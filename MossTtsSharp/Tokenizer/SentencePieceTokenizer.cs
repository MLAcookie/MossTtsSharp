using System.Text.Json;
using SIL.Machine.Tokenization.SentencePiece;
using SpTokenizer = SIL.Machine.Tokenization.SentencePiece.SentencePieceTokenizer;

namespace MossTtsSharp.Tokenizer;

internal class SentencePieceTokenizer : IDisposable
{
    private readonly SpTokenizer _spTokenizer;
    private readonly Dictionary<string, int> _pieceToId;
    private readonly Dictionary<int, string> _idToPiece;
    private readonly int _unkId;

    public SentencePieceTokenizer(string modelPath, string vocabJsonPath)
    {
        _spTokenizer = new SpTokenizer(modelPath);

        var vocabJson = File.ReadAllText(vocabJsonPath);
        _pieceToId = JsonSerializer.Deserialize<Dictionary<string, int>>(vocabJson) ?? new();

        var idToPiecePath = Path.Combine(Path.GetDirectoryName(vocabJsonPath)!, "id_to_piece.json");
        if (File.Exists(idToPiecePath))
        {
            var idJson = File.ReadAllText(idToPiecePath);
            using var doc = JsonDocument.Parse(idJson);
            _idToPiece = new Dictionary<int, string>();
            foreach (var prop in doc.RootElement.EnumerateObject())
            {
                _idToPiece[int.Parse(prop.Name)] = prop.Value.GetString()!;
            }
        }
        else
        {
            _idToPiece = _pieceToId.ToDictionary(kv => kv.Value, kv => kv.Key);
        }

        _unkId = _pieceToId.TryGetValue("<unk>", out var unk) ? unk : 0;
    }

    public int[] Encode(string text)
    {
        var pieces = _spTokenizer.Tokenize(text);
        var ids = new List<int>();
        foreach (var piece in pieces)
        {
            if (_pieceToId.TryGetValue(piece, out var id))
                ids.Add(id);
            else
                ids.Add(_unkId);
        }

        return ids.ToArray();
    }

    public string Decode(int[] ids)
    {
        var pieces = new List<string>();
        foreach (var id in ids)
        {
            if (_idToPiece.TryGetValue(id, out var piece))
                pieces.Add(piece);
        }

        string result = string.Join("", pieces).Replace("▁", " ").Trim();
        return result;
    }

    public void Dispose()
    {
        _spTokenizer.Dispose();
    }
}