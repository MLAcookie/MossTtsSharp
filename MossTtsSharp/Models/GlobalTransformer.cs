using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MossTtsSharp.Backend;

namespace MossTtsSharp.Models;

internal class GlobalTransformer : IDisposable
{
    private readonly InferenceSession _prefillSession;
    private readonly InferenceSession _stepSession;

    private const int NumLayers = Config.MossModelConfig.NumLayers;
    private const int HiddenSize = Config.MossModelConfig.HiddenSize;
    private const int NumHeads = Config.MossModelConfig.NumHeads;
    private const int HeadDim = Config.MossModelConfig.HeadDim;
    private const int LocalPositions = Config.MossModelConfig.LocalPositions;

    private readonly float[][] _kvKeys = new float[NumLayers][];
    private readonly float[][] _kvValues = new float[NumLayers][];
    private int _totalPastLen;

    public int TotalPastLen => _totalPastLen;

    public GlobalTransformer(string ttsModelDir, BackendProvider backend)
    {
        var opts = backend.CreateSessionOptions();
        _prefillSession = new InferenceSession(
            Path.Combine(ttsModelDir, "moss_tts_prefill.onnx"), opts);
        _stepSession = new InferenceSession(
            Path.Combine(ttsModelDir, "moss_tts_decode_step.onnx"), opts);

        for (int i = 0; i < NumLayers; i++)
        {
            _kvKeys[i] = Array.Empty<float>();
            _kvValues[i] = Array.Empty<float>();
        }
    }

    public float[] Prefill(int[][] inputIds, bool[] attentionMask)
    {
        int seqLen = inputIds.Length;
        if (seqLen == 0)
            return Array.Empty<float>();

        int[] flat = new int[seqLen * LocalPositions];
        for (int i = 0; i < seqLen; i++)
            for (int j = 0; j < LocalPositions; j++)
                flat[i * LocalPositions + j] = inputIds[i][j];

        int[] maskInt = new int[seqLen];
        for (int i = 0; i < seqLen; i++)
            maskInt[i] = attentionMask[i] ? 1 : 0;

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids",
                new DenseTensor<int>(flat, [1, seqLen, LocalPositions])),
            NamedOnnxValue.CreateFromTensor("attention_mask",
                new DenseTensor<int>(maskInt, [1, seqLen])),
        };

        using var results = _prefillSession.Run(inputs);
        var hidden = results[0].AsTensor<float>().ToArray();
        _totalPastLen = 0;

        for (int l = 0; l < NumLayers; l++)
        {
            _kvKeys[l] = CopyToBuffer(_kvKeys[l], results[1 + 2 * l].AsTensor<float>());
            _kvValues[l] = CopyToBuffer(_kvValues[l], results[2 + 2 * l].AsTensor<float>());
        }

        _totalPastLen = seqLen;

        float[] lastHidden = new float[HiddenSize];
        Array.Copy(hidden, (seqLen - 1) * HiddenSize, lastHidden, 0, HiddenSize);
        return lastHidden;
    }

    public float[] Step(int[] inputIds)
    {
        int[] pastValidLengths = { _totalPastLen };

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids",
                new DenseTensor<int>(inputIds, [1, 1, LocalPositions])),
            NamedOnnxValue.CreateFromTensor("past_valid_lengths",
                new DenseTensor<int>(pastValidLengths, [1])),
        };

        for (int l = 0; l < NumLayers; l++)
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_{l}",
                new DenseTensor<float>(_kvKeys[l], [1, _totalPastLen, NumHeads, HeadDim])));
            inputs.Add(NamedOnnxValue.CreateFromTensor($"past_value_{l}",
                new DenseTensor<float>(_kvValues[l], [1, _totalPastLen, NumHeads, HeadDim])));
        }

        using var results = _stepSession.Run(inputs);
        var hidden = results[0].AsTensor<float>().ToArray();

        for (int l = 0; l < NumLayers; l++)
        {
            _kvKeys[l] = CopyToBuffer(_kvKeys[l], results[1 + 2 * l].AsTensor<float>());
            _kvValues[l] = CopyToBuffer(_kvValues[l], results[2 + 2 * l].AsTensor<float>());
        }

        _totalPastLen += 1;
        return hidden;
    }

    public void ResetCache()
    {
        for (int i = 0; i < NumLayers; i++)
        {
            _kvKeys[i] = Array.Empty<float>();
            _kvValues[i] = Array.Empty<float>();
        }
        _totalPastLen = 0;
    }

    private static float[] CopyToBuffer(float[] buf, Tensor<float> src)
    {
        int n = (int)src.Length;
        if (buf.Length == n)
        {
            float[] data = src.ToArray();
            Array.Copy(data, 0, buf, 0, n);
            return buf;
        }
        return src.ToArray();
    }

    public void Dispose()
    {
        _prefillSession.Dispose();
        _stepSession.Dispose();
    }
}
