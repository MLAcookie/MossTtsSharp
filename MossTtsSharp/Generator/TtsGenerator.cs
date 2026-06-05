using MossTtsSharp.Config;
using MossTtsSharp.Models;

namespace MossTtsSharp.Generator;

internal class TtsGenerator
{
    private readonly GlobalTransformer _globalTransformer;
    private readonly FrameGenerator _frameGenerator;

    private const int Nvq = MossModelConfig.Nvq;
    private const int H = MossModelConfig.HiddenSize;

    public TtsGenerator(GlobalTransformer globalTransformer, FrameGenerator frameGenerator)
    {
        _globalTransformer = globalTransformer;
        _frameGenerator = frameGenerator;
    }

    public IEnumerable<int[]> GenerateStream(
        int[][] promptInputIds,
        bool[] promptAttentionMask,
        int maxNewFrames = MossModelConfig.MaxNewFrames,
        float? noise = null)
    {
        _globalTransformer.ResetCache();
        _frameGenerator.ResetMask();

        float[] globalHidden = _globalTransformer.Prefill(promptInputIds, promptAttentionMask);
        if (globalHidden.Length == 0) yield break;

        for (int step = 0; step < maxNewFrames; step++)
        {
            var (audioTokens, shouldContinue) = _frameGenerator.GenerateFrame(globalHidden, noise);
            if (!shouldContinue) break;
            int[] genRow = PromptBuilder.BuildGenerationRow(audioTokens);
            yield return audioTokens;
            globalHidden = _globalTransformer.Step(genRow);
        }
    }

    public int[][] Generate(
        int[][] promptInputIds,
        bool[] promptAttentionMask,
        int maxNewFrames = MossModelConfig.MaxNewFrames,
        float? noise = null)
    {
        return GenerateStream(promptInputIds, promptAttentionMask, maxNewFrames, noise).ToArray();
    }
}
