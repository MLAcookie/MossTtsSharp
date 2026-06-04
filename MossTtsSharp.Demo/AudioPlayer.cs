using NAudio.Wave;
using System.Runtime.InteropServices;

namespace MossTtsSharp.Demo;

public static class AudioPlayer
{
    public static void Play(float[] samples, int sampleRate, int channels)
    {
        var format = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, channels);
        var bytes = MemoryMarshal.AsBytes<float>(samples.AsSpan()).ToArray();

        using var ms = new MemoryStream(bytes);
        using var reader = new RawSourceWaveStream(ms, format);
        using var player = new WaveOutEvent();

        player.Init(reader);
        player.Play();

        while (player.PlaybackState == PlaybackState.Playing)
            Thread.Sleep(50);
    }

    public static (Action<float[], int> onChunk, Action finish) CreateStreamingPlayer(
        int sampleRate, int channels)
    {
        var format = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, channels);
        var provider = new BufferedWaveProvider(format)
        {
            BufferDuration = TimeSpan.FromSeconds(10),
            DiscardOnBufferOverflow = false,
        };

        var player = new WaveOutEvent();
        player.Init(provider);
        player.Play();

        // var allSamples = new List<float>();

        Action<float[], int> onChunk = (samples, _) =>
        {
            // allSamples.AddRange(samples);
            var bytes = MemoryMarshal.AsBytes<float>(samples.AsSpan()).ToArray();
            provider.AddSamples(bytes, 0, bytes.Length);
        };

        Action finish = () =>
        {
            while (provider.BufferedDuration.TotalMilliseconds > 50) Thread.Sleep(10);
            Thread.Sleep(100);
            player.Stop();
            player.Dispose();
        };

        return (onChunk, finish);
    }
}