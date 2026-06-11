using Google.Protobuf;

namespace MossTtsSharp.Tokenizer;

internal readonly struct PieceInfo(string piece, float score, int type)
{
    public readonly string Piece = piece;
    public readonly float Score = score;
    public readonly int Type = type;
}

internal readonly struct TrainerSpec(int unknownId, bool byteFallback)
{
    public readonly int UnknownId = unknownId;
    public readonly bool ByteFallback = byteFallback;
}

internal readonly struct NormalizerSpec(
    byte[] precompiledCharsmap,
    bool addDummyPrefix,
    bool removeExtraWhitespaces,
    bool escapeWhitespaces)
{
    public readonly byte[] PrecompiledCharsmap = precompiledCharsmap;
    public readonly bool AddDummyPrefix = addDummyPrefix;
    public readonly bool RemoveExtraWhitespaces = removeExtraWhitespaces;
    public readonly bool EscapeWhitespaces = escapeWhitespaces;
}

internal static class SentencePieceModelParser
{
    public static (PieceInfo[] pieces, TrainerSpec trainerSpec, NormalizerSpec normalizerSpec)
        Parse(string modelPath) => Parse(File.ReadAllBytes(modelPath));

    public static (PieceInfo[] pieces, TrainerSpec trainerSpec, NormalizerSpec normalizerSpec)
        Parse(byte[] data)
    {
        var pieces = new List<PieceInfo>();
        TrainerSpec? trainerSpec = null;
        NormalizerSpec? normalizerSpec = null;

        var input = new CodedInputStream(data);
        while (!input.IsAtEnd)
        {
            uint tag;
            try
            {
                tag = input.ReadTag();
            }
            catch (InvalidProtocolBufferException)
            {
                break;
            }

            var fieldNumber = (int)(tag >> 3);

            switch (fieldNumber)
            {
                case 1: pieces.Add(ParsePiece(input.ReadBytes().ToByteArray())); break;
                case 2: trainerSpec = ParseTrainerSpec(input.ReadBytes().ToByteArray()); break;
                case 3: normalizerSpec = ParseNormalizerSpec(input.ReadBytes().ToByteArray()); break;
                default: input.SkipLastField(); break;
            }
        }

        return (
            pieces.ToArray(),
            trainerSpec ?? new TrainerSpec(0, true),
            normalizerSpec ?? new NormalizerSpec(Array.Empty<byte>(), true, true, true)
        );
    }

    private static PieceInfo ParsePiece(byte[] data)
    {
        var piece = "";
        var score = 0f;
        var type = 1;

        var input = new CodedInputStream(data);
        while (!input.IsAtEnd)
        {
            uint tag;
            try
            {
                tag = input.ReadTag();
            }
            catch (InvalidProtocolBufferException)
            {
                break;
            }

            switch ((int)(tag >> 3))
            {
                case 1: piece = input.ReadString(); break;
                case 2: score = input.ReadFloat(); break;
                case 3: type = input.ReadInt32(); break;
                default: input.SkipLastField(); break;
            }
        }

        return new PieceInfo(piece, score, type);
    }

    private static TrainerSpec ParseTrainerSpec(byte[] data)
    {
        var unknownId = 0;
        var byteFallback = false;

        var input = new CodedInputStream(data);
        while (!input.IsAtEnd)
        {
            uint tag;
            try
            {
                tag = input.ReadTag();
            }
            catch (InvalidProtocolBufferException)
            {
                break;
            }

            switch ((int)(tag >> 3))
            {
                case 26: unknownId = input.ReadInt32(); break;
                case 41: byteFallback = input.ReadBool(); break;
                default: input.SkipLastField(); break;
            }
        }

        return new TrainerSpec(unknownId, byteFallback);
    }

    private static NormalizerSpec ParseNormalizerSpec(byte[] data)
    {
        byte[] precompiledCharsmap = [];
        bool addDummyPrefix = true, removeExtraWhitespaces = true, escapeWhitespaces = true;

        var input = new CodedInputStream(data);
        while (!input.IsAtEnd)
        {
            uint tag;
            try
            {
                tag = input.ReadTag();
            }
            catch (InvalidProtocolBufferException)
            {
                break;
            }

            switch ((int)(tag >> 3))
            {
                case 2: precompiledCharsmap = input.ReadBytes().ToByteArray(); break;
                case 3: addDummyPrefix = input.ReadBool(); break;
                case 4: removeExtraWhitespaces = input.ReadBool(); break;
                case 5: escapeWhitespaces = input.ReadBool(); break;
                default: input.SkipLastField(); break;
            }
        }

        return new NormalizerSpec(precompiledCharsmap, addDummyPrefix,
            removeExtraWhitespaces, escapeWhitespaces);
    }
}