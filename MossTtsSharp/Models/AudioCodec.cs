using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MossTtsSharp.Models;

internal class AudioCodec : IDisposable
{
    private readonly InferenceSession _encoderSession;
    private readonly InferenceSession _decoderSession;

    public AudioCodec(string codecModelDir)
    {
        var opts = new SessionOptions();
        opts.AppendExecutionProvider_CPU();
        _encoderSession = new InferenceSession(
            Path.Combine(codecModelDir, "moss_audio_tokenizer_encode.onnx"), opts);
        _decoderSession = new InferenceSession(
            Path.Combine(codecModelDir, "moss_audio_tokenizer_decode_full.onnx"), opts);
    }

    public (long[] codes, int codeLen) Encode(float[] waveform)
    {
        var channels = Config.MossModelConfig.Channels;
        var audioLen = waveform.Length / channels;
        var downsampleRate = Config.MossModelConfig.DownsampleRate;
        var paddedLen = audioLen;
        var remainder = audioLen % downsampleRate;
        if (remainder != 0)
            paddedLen = audioLen + (downsampleRate - remainder);

        var padded = waveform;
        if (paddedLen != audioLen)
        {
            padded = new float[channels * paddedLen];
            for (var c = 0; c < channels; c++) Array.Copy(waveform, c * audioLen, padded, c * paddedLen, audioLen);
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("waveform",
                new DenseTensor<float>(padded, [1, channels, paddedLen])),
            NamedOnnxValue.CreateFromTensor("input_lengths",
                new DenseTensor<int>(new int[] { audioLen }, [1])),
        };

        using var results = _encoderSession.Run(inputs);
        var codesTensor = results[0].AsTensor<int>();
        var codeLen = codesTensor.Dimensions[1];

        var flatCodes = new long[Config.MossModelConfig.Nvq * codeLen];
        var codesArr = codesTensor.ToArray();
        for (var t = 0; t < codeLen; t++)
        {
            for (var c = 0; c < Config.MossModelConfig.Nvq; c++)
                flatCodes[c * codeLen + t] = codesArr[t * Config.MossModelConfig.Nvq + c];
        }

        return (flatCodes, codeLen);
    }

    public float[] Decode(long[] codes)
    {
        var codeLen = codes.Length / Config.MossModelConfig.Nvq;

        var flatCodesInt = new int[Config.MossModelConfig.Nvq * codeLen];
        for (var t = 0; t < codeLen; t++)
        {
            for (var c = 0; c < Config.MossModelConfig.Nvq; c++)
                flatCodesInt[t * Config.MossModelConfig.Nvq + c] = (int)codes[c * codeLen + t];
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("audio_codes",
                new DenseTensor<int>(flatCodesInt, [1, codeLen, Config.MossModelConfig.Nvq])),
            NamedOnnxValue.CreateFromTensor("audio_code_lengths",
                new DenseTensor<int>(new int[] { codeLen }, [1])),
        };

        using var results = _decoderSession.Run(inputs);
        var waveform = results[0].AsTensor<float>();
        var audioLen = results[1].AsTensor<int>().ToArray()[0];
        var fullLen = waveform.Dimensions[2];
        var flat = waveform.ToArray();

        var interleaved = new float[audioLen * Config.MossModelConfig.Channels];
        for (var t = 0; t < audioLen; t++)
        {
            for (var c = 0; c < Config.MossModelConfig.Channels; c++)
                interleaved[t * Config.MossModelConfig.Channels + c] = flat[c * fullLen + t];
        }

        return interleaved;
    }

    public void Dispose()
    {
        _encoderSession.Dispose();
        _decoderSession.Dispose();
    }
}