using Google.Protobuf;

namespace MossTtsSharp.Tokenizer;

internal readonly struct PieceInfo
{
    public readonly string Piece;
    public readonly float Score;
    public readonly int Type;

    public PieceInfo(string piece, float score, int type)
    {
        Piece = piece;
        Score = score;
        Type = type;
    }
}

internal readonly struct TrainerSpec
{
    public readonly int UnknownId;
    public readonly bool ByteFallback;

    public TrainerSpec(int unknownId, bool byteFallback)
    {
        UnknownId = unknownId;
        ByteFallback = byteFallback;
    }
}

internal readonly struct NormalizerSpec
{
    public readonly byte[] PrecompiledCharsmap;
    public readonly bool AddDummyPrefix;
    public readonly bool RemoveExtraWhitespaces;
    public readonly bool EscapeWhitespaces;

    public NormalizerSpec(
        byte[] precompiledCharsmap, bool addDummyPrefix,
        bool removeExtraWhitespaces, bool escapeWhitespaces)
    {
        PrecompiledCharsmap = precompiledCharsmap;
        AddDummyPrefix = addDummyPrefix;
        RemoveExtraWhitespaces = removeExtraWhitespaces;
        EscapeWhitespaces = escapeWhitespaces;
    }
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

            int fieldNumber = (int)(tag >> 3);

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
        string piece = "";
        float score = 0f;
        int type = 1;

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
        int unknownId = 0;
        bool byteFallback = false;

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
        byte[] precompiledCharsmap = Array.Empty<byte>();
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