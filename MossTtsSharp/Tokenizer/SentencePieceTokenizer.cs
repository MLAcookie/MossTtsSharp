using System.Text;

namespace MossTtsSharp.Tokenizer;

internal class SentencePieceTokenizer : IDisposable
{
    private readonly Dictionary<string, int> _pieceToId;
    private readonly float[] _pieceScores;
    private readonly int _unknownId;
    private readonly bool _addDummyPrefix;
    private readonly bool _removeExtraWhitespaces;
    private readonly bool _escapeWhitespaces;

    public SentencePieceTokenizer(string modelPath)
    {
        var (pieces, trainerSpec, normalizerSpec) = SentencePieceModelParser.Parse(modelPath);

        _pieceToId = new Dictionary<string, int>(pieces.Length);
        _pieceScores = new float[pieces.Length];
        for (var i = 0; i < pieces.Length; i++)
        {
            _pieceToId[pieces[i].Piece] = i;
            _pieceScores[i] = pieces[i].Score;
        }

        _unknownId = trainerSpec.UnknownId;
        _addDummyPrefix = normalizerSpec.AddDummyPrefix;
        _removeExtraWhitespaces = normalizerSpec.RemoveExtraWhitespaces;
        _escapeWhitespaces = normalizerSpec.EscapeWhitespaces;
    }

    public int[] Encode(string text)
    {
        var normalized = Normalize(text);
        var symbols = TokenizeToSymbols(normalized);
        BpeMerge(symbols);

        var ids = new int[symbols.Count];
        for (var i = 0; i < symbols.Count; i++)
            ids[i] = _pieceToId.GetValueOrDefault(symbols[i], _unknownId);
        return ids;
    }

    private string Normalize(string text)
    {
        var sb = new StringBuilder(text.Length);
        var lastWasWhitespace = false;

        foreach (var c in text.Normalize(NormalizationForm.FormKC))
        {
            var mapped = MapChar(c);

            if (mapped == ' ' || mapped == '\n' || mapped == '\r' || mapped == '\t')
            {
                if (_removeExtraWhitespaces)
                {
                    if (!lastWasWhitespace) sb.Append(' ');
                    lastWasWhitespace = true;
                }
                else
                {
                    sb.Append(mapped);
                }
            }
            else
            {
                sb.Append(mapped);
                lastWasWhitespace = false;
            }
        }

        var result = sb.ToString().Trim();
        if (_escapeWhitespaces) result = result.Replace(" ", "▁");
        if (_addDummyPrefix && result.Length > 0 && !result.StartsWith('▁')) result = "▁" + result;

        return result;
    }

    private static char MapChar(char c)
    {
        if (c >= 0xFF01 && c <= 0xFF5E)
            return (char)(c - 0xFF01 + 0x21);
        if (c == 0x3000)
            return ' ';
        if (c is '\n' or '\r' or '\t')
            return ' ';
        return c;
    }

    private List<string> TokenizeToSymbols(string normalized)
    {
        var symbols = new List<string>();
        var i = 0;
        while (i < normalized.Length)
        {
            var matchLength = TryMatchSpecialToken(normalized, i);
            if (matchLength > 0)
            {
                symbols.Add(normalized.Substring(i, matchLength));
                i += matchLength;
                continue;
            }

            int codePoint;
            if (char.IsHighSurrogate(normalized[i]) && i + 1 < normalized.Length &&
                char.IsLowSurrogate(normalized[i + 1]))
            {
                codePoint = char.ConvertToUtf32(normalized[i], normalized[i + 1]);
                i += 2;
            }
            else
            {
                codePoint = normalized[i];
                i++;
            }

            var charStr = char.ConvertFromUtf32(codePoint);

            if (_pieceToId.ContainsKey(charStr))
            {
                symbols.Add(charStr);
            }
            else
            {
                var bytes = Encoding.UTF8.GetBytes(charStr);
                symbols.AddRange(bytes.Select(b => $"<0x{b:X2}>"));
            }
        }

        return symbols;
    }

    private int TryMatchSpecialToken(string text, int startPos)
    {
        if (text[startPos] != '<') return 0;
        for (var end = startPos + 1; end <= text.Length && end - startPos <= 20; end++)
        {
            var sub = text.Substring(startPos, end - startPos);
            if (_pieceToId.TryGetValue(sub, out var id) && id != _unknownId && id <= 14)
                return end - startPos;
        }

        return 0;
    }

    private void BpeMerge(List<string> symbols)
    {
        while (true)
        {
            var bestPos = -1;
            var bestScore = float.NegativeInfinity;

            for (var i = 0; i < symbols.Count - 1; i++)
            {
                var pair = symbols[i] + symbols[i + 1];
                if (_pieceToId.TryGetValue(pair, out var id) && id != _unknownId)
                {
                    var score = _pieceScores[id];
                    if (score > bestScore)
                    {
                        bestPos = i;
                        bestScore = score;
                    }
                }
            }

            if (bestPos < 0) break;
            symbols[bestPos] = symbols[bestPos] + symbols[bestPos + 1];
            symbols.RemoveAt(bestPos + 1);
        }
    }

    public void Dispose()
    {
    }
}