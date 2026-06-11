using MossTtsSharp.Audio;
using MossTtsSharp.Config;
using MossTtsSharp.Generator;
using MossTtsSharp.Models;
using MossTtsSharp.Tokenizer;

namespace MossTtsSharp.Pipeline;

public class MossTtsPipeline : IDisposable, IAsyncDisposable
{
    private readonly GlobalTransformer _globalTransformer;
    private readonly FrameGenerator _frameGenerator;
    private readonly AudioCodec _audioCodec;
    private readonly TtsGenerator _generator;
    private readonly PromptBuilder _promptBuilder;
    private readonly string _codecModelDir;
    private bool _disposed;

    private MossTtsPipeline(
        GlobalTransformer globalTransformer,
        FrameGenerator frameGenerator,
        AudioCodec audioCodec,
        TtsGenerator generator,
        PromptBuilder promptBuilder,
        string codecModelDir
    )
    {
        _globalTransformer = globalTransformer;
        _frameGenerator = frameGenerator;
        _audioCodec = audioCodec;
        _generator = generator;
        _promptBuilder = promptBuilder;
        _codecModelDir = codecModelDir;
    }

    public static async Task<MossTtsPipeline> CreateAsync(
        MossConfig config,
        CancellationToken ct = default
    )
    {
        var ttsDir = config.ResolveTtsModelDir();
        var codecDir = config.ResolveCodecModelDir();
        var tkPath = config.ResolveTokenizerModelFile();

        GlobalTransformer? global = null;
        FrameGenerator? frame = null;
        AudioCodec? codec = null;
        SentencePieceTokenizer? tokenizer = null;

        try
        {
            var backend = config.Device;
            var globalTask = Task.Run(() => new GlobalTransformer(ttsDir, backend), ct);
            var frameTask = Task.Run(() => new FrameGenerator(ttsDir, backend), ct);
            var codecTask = Task.Run(() => new AudioCodec(codecDir), ct);
            var tkTask = Task.Run(() => new SentencePieceTokenizer(tkPath), ct);

            await Task.WhenAll(globalTask, frameTask, codecTask, tkTask);
            ct.ThrowIfCancellationRequested();

            global = globalTask.Result;
            frame = frameTask.Result;
            codec = codecTask.Result;
            tokenizer = tkTask.Result;

            var generator = new TtsGenerator(global, frame);
            var promptBuilder = new PromptBuilder(tokenizer);

            return new MossTtsPipeline(global, frame, codec, generator, promptBuilder, codecDir);
        }
        catch
        {
            global?.Dispose();
            frame?.Dispose();
            codec?.Dispose();
            tokenizer?.Dispose();
            throw;
        }
    }

    private (int[][] promptIds, bool[] promptMask) PreparePrompt(
        string text,
        string promptAudioPath
    )
    {
        var (refSamples, refSr, refCh) = AudioFile.Read(promptAudioPath);
        var ref48k = refSamples;
        if (refSr != MossModelConfig.SampleRate)
            ref48k = AudioResampler.Resample(refSamples, refSr, refCh, MossModelConfig.SampleRate);

        var audioLenMono = ref48k.Length / refCh;
        var refStereo = new float[audioLenMono * MossModelConfig.Channels];
        var isMonoInput = refCh == 1;

        if (isMonoInput)
        {
            for (var t = 0; t < audioLenMono; t++)
            {
                refStereo[t] = ref48k[t];
                refStereo[t + audioLenMono] = ref48k[t];
            }
        }
        else
        {
            for (var t = 0; t < audioLenMono; t++)
            {
                refStereo[t] = ref48k[t * 2];
                refStereo[t + audioLenMono] = ref48k[t * 2 + 1];
            }
        }

        var (encodedCodes, codeLen) = _audioCodec.Encode(refStereo);

        var promptAudioCodes = new int[codeLen][];
        for (var t = 0; t < codeLen; t++)
        {
            promptAudioCodes[t] = new int[MossModelConfig.Nvq];
            for (var c = 0; c < MossModelConfig.Nvq; c++)
                promptAudioCodes[t][c] = (int)encodedCodes[c * codeLen + t];
        }

        return _promptBuilder.BuildVoiceClone(text, promptAudioCodes);
    }

    public (float[] waveform, int sampleRate) Synthesize(
        string text,
        string promptAudioPath,
        float? noise = null
    )
    {
        var (promptIds, promptMask) = PreparePrompt(text, promptAudioPath);
        var audioTokens = _generator.Generate(promptIds, promptMask, noise: noise);

        if (audioTokens.Length == 0)
            return (Array.Empty<float>(), MossModelConfig.SampleRate);

        var numFrames = audioTokens.Length;
        var decodeCodes = new long[MossModelConfig.Nvq * numFrames];
        for (var t = 0; t < numFrames; t++)
        {
            for (var c = 0; c < MossModelConfig.Nvq; c++)
                decodeCodes[c * numFrames + t] = audioTokens[t][c];
        }

        var output = _audioCodec.Decode(decodeCodes);
        return (output, MossModelConfig.SampleRate);
    }

    public Task<(float[] waveform, int sampleRate)> SynthesizeAsync(
        string text,
        string promptAudioPath,
        float? noise = null,
        CancellationToken ct = default
    )
    {
        return Task.Run(
            () =>
            {
                ct.ThrowIfCancellationRequested();
                return Synthesize(text, promptAudioPath, noise);
            },
            ct
        );
    }

    public void SynthesizeStream(
        string text,
        string promptAudioPath,
        Action<float[]> onAudioChunk,
        float? noise = null
    )
    {
        var (promptIds, promptMask) = PreparePrompt(text, promptAudioPath);
        using var decoder = new StreamingDecoder(_codecModelDir);

        foreach (var frame in _generator.GenerateStream(promptIds, promptMask, noise: noise))
        {
            var audio = decoder.DecodeFrame(frame);
            if (audio.Length > 0)
                onAudioChunk(audio);
        }

        var remaining = decoder.Flush();
        if (remaining.Length > 0)
            onAudioChunk(remaining);
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;

        _globalTransformer.Dispose();
        _frameGenerator.Dispose();
        _audioCodec.Dispose();
        _promptBuilder.Dispose();

        GC.SuppressFinalize(this);
    }

    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }
}
