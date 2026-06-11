using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MossTtsSharp.Backend;

namespace MossTtsSharp.Models;

/// <summary>
/// Generates one full frame of tokens (1 text + 16 audio) from global_hidden,
/// using the official local_fixed_sampled_frame ONNX model.
/// Sampling and repetition penalty are handled inside the ONNX graph.
/// </summary>
internal class FrameGenerator : IDisposable
{
    private readonly InferenceSession _session;
    private readonly int[,] _repetitionSeenMask;
    private readonly Random _rng;

    private const int Nvq = Config.MossModelConfig.Nvq;
    private const int CodebookSize = Config.MossModelConfig.CodebookSize;

    public FrameGenerator(string ttsModelDir, BackendProvider backend)
    {
        var opts = backend.CreateSessionOptions();
        _session = new InferenceSession(
            Path.Combine(ttsModelDir, "moss_tts_local_fixed_sampled_frame.onnx"), opts);

        _repetitionSeenMask = new int[Nvq, CodebookSize];
        _rng = new Random();
    }

    public (int[] audioTokens, bool shouldContinue) GenerateFrame(float[] globalHidden, float? noise = null)
    {
        var assistantRandomU = noise ?? (float)_rng.NextDouble();
        var audioRandomU = new float[Nvq];
        for (var i = 0; i < Nvq; i++)
            audioRandomU[i] = noise ?? (float)_rng.NextDouble();

        var maskFlat = new int[Nvq * CodebookSize];
        for (var c = 0; c < Nvq; c++)
        {
            for (var k = 0; k < CodebookSize; k++)
                maskFlat[c * CodebookSize + k] = _repetitionSeenMask[c, k];
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("global_hidden",
                new DenseTensor<float>(globalHidden, [1, Config.MossModelConfig.HiddenSize])),
            NamedOnnxValue.CreateFromTensor("repetition_seen_mask",
                new DenseTensor<int>(maskFlat, [1, Nvq, CodebookSize])),
            NamedOnnxValue.CreateFromTensor("assistant_random_u",
                new DenseTensor<float>(new[] { assistantRandomU }, [1])),
            NamedOnnxValue.CreateFromTensor("audio_random_u",
                new DenseTensor<float>(audioRandomU, [1, Nvq])),
        };

        using var results = _session.Run(inputs);
        var shouldContinue = results[0].AsTensor<int>().ToArray()[0] != 0;
        var tokens = results[1].AsTensor<int>().ToArray();

        for (var c = 0; c < Nvq; c++)
        {
            var token = tokens[c];
            if (token >= 0 && token < CodebookSize) _repetitionSeenMask[c, token] = 1;
        }

        return (tokens, shouldContinue);
    }

    public void ResetMask()
    {
        Array.Clear(_repetitionSeenMask);
    }

    public void Dispose() => _session.Dispose();
}