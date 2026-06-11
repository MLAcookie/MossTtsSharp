using MossTtsSharp.Config;
using MossTtsSharp.Tokenizer;

namespace MossTtsSharp.Generator;

internal class PromptBuilder(SentencePieceTokenizer tokenizer) : IDisposable
{
    private const string UserRolePrefix = "user\n";
    private const string UserTemplateReferencePrefix = "<user_inst>\n- Reference(s):\n";
    private const string UserTemplateAfterReference =
        "\n- Instruction:\nNone\n" +
        "- Tokens:\nNone\n" +
        "- Quality:\nNone\n" +
        "- Sound Event:\nNone\n" +
        "- Ambient Sound:\nNone\n" +
        "- Language:\nNone\n" +
        "- Text:\n";
    private const string UserTemplateSuffix = "</user_inst>";
    private const string AssistantRolePrefix = "assistant\n";

    private int[] Encode(string text) => tokenizer.Encode(text);

    private static int[] BuildTextRow(int textTokenId)
    {
        var row = new int[MossModelConfig.LocalPositions];
        for (var i = 0; i < MossModelConfig.LocalPositions; i++)
            row[i] = MossModelConfig.AudioPadTokenId;
        row[0] = textTokenId;
        return row;
    }

    private static List<int[]> BuildTextRows(IEnumerable<int> tokenIds)
    {
        return tokenIds.Select(BuildTextRow).ToList();
    }

    private static List<int[]> BuildAudioPrefixRows(int[][] promptAudioCodes, int slotTokenId)
    {
        var frames = promptAudioCodes.Length;
        var rows = new List<int[]>(frames);
        for (var t = 0; t < frames; t++)
        {
            var row = new int[MossModelConfig.LocalPositions];
            for (var i = 0; i < MossModelConfig.LocalPositions; i++)
                row[i] = MossModelConfig.AudioPadTokenId;
            row[0] = slotTokenId;
            for (var c = 0; c < MossModelConfig.Nvq; c++)
                row[c + 1] = promptAudioCodes[t][c];
            rows.Add(row);
        }
        return rows;
    }

    private List<int> BuildUserPromptPrefix()
    {
        return Encode(UserRolePrefix)
            .Concat(Encode(UserTemplateReferencePrefix))
            .ToList();
    }

    private List<int> BuildUserPromptAfterReference()
    {
        return Encode(UserTemplateAfterReference).ToList();
    }

    private List<int> BuildAssistantPromptPrefix()
    {
        return Encode(UserTemplateSuffix)
            .Concat([MossModelConfig.ImEndTokenId])
            .Concat([MossModelConfig.ImStartTokenId])
            .Concat(Encode(AssistantRolePrefix))
            .ToList();
    }

    public (int[][] inputIds, bool[] attentionMask) BuildVoiceClone(
        string text,
        int[][] promptAudioCodes)
    {
        var textTokenIds = Encode(text);

        var promptTokenIds = BuildUserPromptPrefix()
            .Concat([MossModelConfig.AudioStartTokenId])
            .ToList();

        var suffixTokenIds = new List<int> { MossModelConfig.AudioEndTokenId }
            .Concat(BuildUserPromptAfterReference())
            .Concat(textTokenIds)
            .Concat(BuildAssistantPromptPrefix())
            .Concat([MossModelConfig.AudioStartTokenId])
            .ToList();

        var rows = new List<int[]>();
        rows.AddRange(BuildTextRows(promptTokenIds));
        rows.AddRange(BuildAudioPrefixRows(promptAudioCodes, MossModelConfig.AudioUserSlotTokenId));
        rows.AddRange(BuildTextRows(suffixTokenIds));

        var inputIds = rows.ToArray();
        var attentionMask = new bool[inputIds.Length];
        Array.Fill(attentionMask, true);
        return (inputIds, attentionMask);
    }

    public (int[][] inputIds, bool[] attentionMask) BuildContinuation(string text)
    {
        var textTokenIds = Encode(text);

        var promptTokenIds = BuildUserPromptPrefix()
            .Concat(Encode("None"))
            .Concat(BuildUserPromptAfterReference())
            .Concat(textTokenIds)
            .Concat(BuildAssistantPromptPrefix())
            .ToList();

        var rows = new List<int[]>();
        rows.AddRange(BuildTextRows(promptTokenIds));
        rows.AddRange(BuildTextRows([MossModelConfig.AudioStartTokenId]));

        var inputIds = rows.ToArray();
        var attentionMask = new bool[inputIds.Length];
        Array.Fill(attentionMask, true);
        return (inputIds, attentionMask);
    }

    public static int[] BuildGenerationRow(int[] audioTokenIds)
    {
        var row = new int[MossModelConfig.LocalPositions];
        for (var i = 0; i < MossModelConfig.LocalPositions; i++)
            row[i] = MossModelConfig.AudioPadTokenId;
        row[0] = MossModelConfig.AudioAssistantSlotTokenId;
        for (var c = 0; c < Math.Min(audioTokenIds.Length, MossModelConfig.Nvq); c++)
            row[c + 1] = audioTokenIds[c];
        return row;
    }

    public void Dispose() => tokenizer.Dispose();
}
