using MossTtsSharp.Backend;

namespace MossTtsSharp.Config;

/// <summary>
/// Resolves model paths with priority: explicit property > environment variable > throw.
/// Expected directory layout:
///   {ModelsRoot}/
///     MOSS-TTS-Nano-100M-ONNX/       (tts models + tokenizer.model)
///     MOSS-Audio-Tokenizer-Nano-ONNX/ (codec models)
/// </summary>
public class MossConfig
{
    public string? TtsModelDir { get; set; }
    public string? CodecModelDir { get; set; }
    public string? TokenizerModelFile { get; set; }
    public string? VocabJsonFile { get; set; }

    public BackendProvider Device { get; set; }

    public string? ModelsRoot { get; set; }

    public string ResolveTtsModelDir()
    {
        if (!string.IsNullOrWhiteSpace(TtsModelDir)) return TtsModelDir;
        var root = ModelsRoot ?? GetEnv("MOSSTTS_MODELS_DIR");
        if (root == null) throw NewException("ModelsRoot or MOSSTTS_MODELS_DIR environment variable");
        return Path.Combine(root, "MOSS-TTS-Nano-100M-ONNX");
    }

    public string ResolveCodecModelDir()
    {
        if (!string.IsNullOrWhiteSpace(CodecModelDir)) return CodecModelDir;
        var root = ModelsRoot ?? GetEnv("MOSSTTS_MODELS_DIR");
        if (root == null) throw NewException("ModelsRoot or MOSSTTS_MODELS_DIR environment variable");
        return Path.Combine(root, "MOSS-Audio-Tokenizer-Nano-ONNX");
    }

    public string ResolveTokenizerModelFile()
    {
        if (!string.IsNullOrWhiteSpace(TokenizerModelFile))
            return TokenizerModelFile;
        var ttsDir = ResolveTtsModelDir();
        return Path.Combine(ttsDir, "tokenizer.model");
    }

    public string ResolveVocabJsonFile()
    {
        if (!string.IsNullOrWhiteSpace(VocabJsonFile)) return VocabJsonFile;
        var ttsDir = ResolveTtsModelDir();
        var path = Path.Combine(ttsDir, "vocab.json");
        if (File.Exists(path)) return path;
        var root = ModelsRoot ?? GetEnv("MOSSTTS_MODELS_DIR");
        if (root != null)
        {
            path = Path.Combine(root, "vocab.json");
            if (File.Exists(path)) return path;
        }

        throw new InvalidOperationException(
            "vocab.json not found. Set VocabJsonFile explicitly, or place vocab.json in the TTS model directory.");
    }

    private static string? GetEnv(string name) =>
        Environment.GetEnvironmentVariable(name);

    private static Exception NewException(string what) =>
        new InvalidOperationException($"Model path not configured. Set {what}.");
}