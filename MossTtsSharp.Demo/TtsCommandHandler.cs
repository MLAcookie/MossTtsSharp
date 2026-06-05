using MossTtsSharp.Audio;
using MossTtsSharp.Config;
using MossTtsSharp.Cpu;
using MossTtsSharp.Pipeline;

namespace MossTtsSharp.Demo;

public static class TtsCommandHandler
{
    public static async Task<int> ExecuteSynAsync(
        string text, FileInfo prompt, FileInfo? output, DirectoryInfo? modelsDir, float? noise)
    {
        if (noise.HasValue)
            Console.WriteLine($"Noise: {noise.Value:F4} (deterministic output)");

        try
        {
            var config = new MossConfig { Device = new CpuBackend() };
            if (modelsDir != null) config.ModelsRoot = modelsDir.FullName;

            LogMode(text, prompt.FullName, output?.FullName);

            await using var pipeline = await MossTtsPipeline.CreateAsync(config);
            var sw = System.Diagnostics.Stopwatch.StartNew();

            var (waveform, sampleRate) = pipeline.Synthesize(text, prompt.FullName, noise);
            sw.Stop();
            Console.WriteLine(
                $"Generated: {waveform.Length / (float)sampleRate / MossModelConfig.Channels:F2}s in {sw.Elapsed.TotalSeconds:F1}s");

            if (output != null)
            {
                AudioFile.Write(output.FullName, waveform, sampleRate, MossModelConfig.Channels);
                Console.WriteLine($"Saved: {output.FullName}");
            }
            else
            {
                Console.WriteLine("Playing...");
                AudioPlayer.Play(waveform, sampleRate, MossModelConfig.Channels);
            }

            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    public static async Task<int> ExecuteStreamAsync(
        string text, FileInfo prompt, FileInfo? output, DirectoryInfo? modelsDir, float? noise)
    {
        if (noise.HasValue)
            Console.WriteLine($"Noise: {noise.Value:F4} (deterministic output)");

        try
        {
            var config = new MossConfig { Device = new CpuBackend() };
            if (modelsDir != null) config.ModelsRoot = modelsDir.FullName;

            LogMode(text, prompt.FullName, output?.FullName);

            await using var pipeline = await MossTtsPipeline.CreateAsync(config);
            var sw = System.Diagnostics.Stopwatch.StartNew();

            if (output != null)
            {
                var allSamples = new List<float>();
                int frameCount = 0;

                pipeline.SynthesizeStream(text, prompt.FullName, audio =>
                {
                    allSamples.AddRange(audio);
                    frameCount++;
                    Console.Write(".");
                }, noise);

                sw.Stop();
                Console.WriteLine();
                float duration = allSamples.Count / (float)MossModelConfig.SampleRate / MossModelConfig.Channels;
                Console.WriteLine($"Generated: {duration:F2}s in {frameCount} frames, {sw.Elapsed.TotalSeconds:F1}s");

                AudioFile.Write(output.FullName, allSamples.ToArray(), MossModelConfig.SampleRate, MossModelConfig.Channels);
                Console.WriteLine($"Saved: {output.FullName}");
            }
            else
            {
                Console.WriteLine("Streaming... ");

                var (onChunk, finish) =
                    AudioPlayer.CreateStreamingPlayer(MossModelConfig.SampleRate, MossModelConfig.Channels);

                pipeline.SynthesizeStream(text, prompt.FullName, audio =>
                {
                    onChunk(audio, 0);
                    Console.Write(".");
                }, noise);

                finish();
                sw.Stop();
                Console.WriteLine($" Done in {sw.Elapsed.TotalSeconds:F1}s");
            }

            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static void LogMode(string text, string promptFile, string? outputPath)
    {
        Console.WriteLine($"Prompt: {Path.GetFileName(promptFile)}");
        Console.WriteLine($"Text:   {text}");
        Console.WriteLine(outputPath != null ? $"Output: {outputPath}" : "Output: playback");
        Console.WriteLine();
    }
}
