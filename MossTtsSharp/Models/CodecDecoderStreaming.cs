using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MossTtsSharp.Models;

internal class CodecDecoderStreaming : IDisposable
{
    private readonly InferenceSession _session;

    private static readonly int[] _contexts = { 500, 500, 500, 500, 800, 800, 1200, 1200, 1600, 1600, 1600, 1600 };
    private const int NumHeads = 4;
    private const int HeadDim = 64;
    private const int Nvq = Config.MossModelConfig.Nvq;

    private readonly float[][] _cachedKeys = new float[12][];
    private readonly float[][] _cachedValues = new float[12][];
    private readonly int[][] _cachedPositions = new int[12][];
    private readonly int[] _attnOffsets = new int[12];
    private readonly int[] _xfmrOffsets = new int[4];
    private readonly List<int> _codeBuffer = new();
    private int _batchesDecoded;

    public CodecDecoderStreaming(string codecModelDir)
    {
        var opts = new SessionOptions();
        opts.AppendExecutionProvider_CPU();
        _session = new InferenceSession(
            Path.Combine(codecModelDir, "moss_audio_tokenizer_decode_step.onnx"), opts);

        for (int l = 0; l < 12; l++)
        {
            int ctx = _contexts[l];
            _cachedKeys[l] = new float[1 * NumHeads * ctx * HeadDim];
            _cachedValues[l] = new float[1 * NumHeads * ctx * HeadDim];
            _cachedPositions[l] = new int[1 * ctx];
            Array.Fill(_cachedPositions[l], -1);
        }
    }

    private int Budget => _batchesDecoded switch
    {
        <= 0 => 1,
        <= 2 => 2,
        <= 5 => 4,
        _ => 8
    };

    public float[] DecodeFrame(int[] tokens)
    {
        _codeBuffer.AddRange(tokens);
        int required = Budget * Nvq;
        if (_codeBuffer.Count < required)
            return Array.Empty<float>();

        return DecodeAccumulated();
    }

    public float[] Flush()
    {
        if (_codeBuffer.Count == 0)
            return Array.Empty<float>();
        return DecodeAccumulated();
    }

    private float[] DecodeAccumulated()
    {
        int frameCount = _codeBuffer.Count / Nvq;
        int[] flatCodes = _codeBuffer.ToArray();
        _codeBuffer.Clear();
        _batchesDecoded++;

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("audio_codes",
                new DenseTensor<int>(flatCodes, [1, frameCount, Nvq])),
            NamedOnnxValue.CreateFromTensor("audio_code_lengths",
                new DenseTensor<int>(new int[] { frameCount }, [1])),
        };

        for (int t = 0; t < 4; t++)
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor($"transformer_offset_{t}",
                new DenseTensor<int>(new[] { _xfmrOffsets[t] }, [1])));
        }

        for (int l = 0; l < 12; l++)
        {
            int ctx = _contexts[l];
            inputs.Add(NamedOnnxValue.CreateFromTensor($"attn_offset_{l}",
                new DenseTensor<int>(new[] { _attnOffsets[l] }, [1])));
            inputs.Add(NamedOnnxValue.CreateFromTensor($"attn_cached_keys_{l}",
                new DenseTensor<float>(_cachedKeys[l], [1, NumHeads, ctx, HeadDim])));
            inputs.Add(NamedOnnxValue.CreateFromTensor($"attn_cached_values_{l}",
                new DenseTensor<float>(_cachedValues[l], [1, NumHeads, ctx, HeadDim])));
            inputs.Add(NamedOnnxValue.CreateFromTensor($"attn_cached_positions_{l}",
                new DenseTensor<int>(_cachedPositions[l], [1, ctx])));
        }

        using var results = _session.Run(inputs);

        var audio = results[0].AsTensor<float>();
        int audioLength = results[1].AsTensor<int>().ToArray()[0];

        int outIdx = 2;
        for (int t = 0; t < 4; t++)
            _xfmrOffsets[t] = results[outIdx++].AsTensor<int>().ToArray()[0];

        for (int l = 0; l < 12; l++)
        {
            _attnOffsets[l] = results[outIdx++].AsTensor<int>().ToArray()[0];
            _cachedKeys[l] = CopyToBuffer(_cachedKeys[l], results[outIdx++].AsTensor<float>());
            _cachedValues[l] = CopyToBuffer(_cachedValues[l], results[outIdx++].AsTensor<float>());
            _cachedPositions[l] = CopyToBufferInt(_cachedPositions[l], results[outIdx++].AsTensor<int>());
        }

        if (audioLength <= 0)
            return Array.Empty<float>();

        int channels = Config.MossModelConfig.Channels;
        float[] flat = audio.ToArray();
        float[] interleaved = new float[audioLength * channels];

        for (int t = 0; t < audioLength; t++)
            for (int c = 0; c < channels; c++)
                interleaved[t * channels + c] = flat[c * audioLength + t];

        return interleaved;
    }

    public void ResetCache()
    {
        _codeBuffer.Clear();
        _batchesDecoded = 0;
        for (int l = 0; l < 12; l++)
        {
            int ctx = _contexts[l];
            int size = 1 * NumHeads * ctx * HeadDim;
            if (_cachedKeys[l].Length != size) _cachedKeys[l] = new float[size];
            else Array.Clear(_cachedKeys[l]);
            if (_cachedValues[l].Length != size) _cachedValues[l] = new float[size];
            else Array.Clear(_cachedValues[l]);
            _attnOffsets[l] = 0;
            int posSize = 1 * ctx;
            if (_cachedPositions[l].Length != posSize) _cachedPositions[l] = new int[posSize];
            Array.Fill(_cachedPositions[l], -1);
        }
        for (int t = 0; t < 4; t++)
            _xfmrOffsets[t] = 0;
    }

    private static float[] CopyToBuffer(float[] buf, Tensor<float> src)
    {
        int n = (int)src.Length;
        if (buf.Length < n) return src.ToArray();
        float[] data = src.ToArray();
        Array.Copy(data, 0, buf, 0, n);
        return buf;
    }

    private static int[] CopyToBufferInt(int[] buf, Tensor<int> src)
    {
        int n = (int)src.Length;
        if (buf.Length < n) return src.ToArray();
        int[] data = src.ToArray();
        Array.Copy(data, 0, buf, 0, n);
        return buf;
    }

    public void Dispose() => _session.Dispose();
}
