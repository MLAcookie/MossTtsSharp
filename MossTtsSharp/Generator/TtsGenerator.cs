using MossTtsSharp.Config;
using MossTtsSharp.Models;

namespace MossTtsSharp.Generator;

internal class TtsGenerator(GlobalTransformer globalTransformer, FrameGenerator frameGenerator)
{
    public IEnumerable<int[]> GenerateStream(
        int[][] promptInputIds,
        bool[] promptAttentionMask,
        int maxNewFrames = MossModelConfig.MaxNewFrames,
        float? noise = null)
    {
        globalTransformer.ResetCache();
        frameGenerator.ResetMask();

        var globalHidden = globalTransformer.Prefill(promptInputIds, promptAttentionMask);
        if (globalHidden.Length == 0) yield break;

        for (var step = 0; step < maxNewFrames; step++)
        {
            var (audioTokens, shouldContinue) = frameGenerator.GenerateFrame(globalHidden, noise);
            if (!shouldContinue) break;
            var genRow = PromptBuilder.BuildGenerationRow(audioTokens);
            yield return audioTokens;
            globalHidden = globalTransformer.Step(genRow);
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
