namespace MossTtsSharp.Audio;

/// <summary>
/// Audio file I/O using pure C# WAV handling.
/// Reads 8/16/32-bit PCM WAV files, writes 16-bit PCM WAV.
/// No external audio library dependencies.
/// </summary>
public static class AudioFile
{
    public static (float[] samples, int sampleRate, int channels) Read(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        if (reader.ReadInt32() != 0x46464952)
            throw new InvalidDataException("Not a valid WAV file");
        reader.ReadInt32();
        if (reader.ReadInt32() != 0x45564157)
            throw new InvalidDataException("Not a valid WAV file");
        if (reader.ReadInt32() != 0x20746D66)
            throw new InvalidDataException("fmt chunk not found");

        var fmtSize = reader.ReadInt32();
        reader.ReadInt16();
        var channels = reader.ReadInt16();
        var sampleRate = reader.ReadInt32();
        reader.ReadInt32();
        reader.ReadInt16();
        var bitsPerSample = reader.ReadInt16();

        if (fmtSize > 16)
            stream.Seek(fmtSize - 16, SeekOrigin.Current);

        while (stream.Position < stream.Length)
        {
            var chunkId = reader.ReadInt32();
            var chunkSize = reader.ReadInt32();

            if (chunkId == 0x61746164)
            {
                var bytesPerSample = bitsPerSample / 8;
                var sampleCount = chunkSize / bytesPerSample;
                var samples = new float[sampleCount];

                if (bitsPerSample == 16)
                {
                    for (var i = 0; i < sampleCount; i++)
                        samples[i] = reader.ReadInt16() / 32768f;
                }
                else if (bitsPerSample == 32)
                {
                    for (var i = 0; i < sampleCount; i++)
                        samples[i] = reader.ReadSingle();
                }
                else if (bitsPerSample == 8)
                {
                    for (var i = 0; i < sampleCount; i++)
                        samples[i] = (reader.ReadByte() - 128) / 128f;
                }
                else
                {
                    throw new NotSupportedException($"Unsupported bit depth: {bitsPerSample}");
                }

                return (samples, sampleRate, channels);
            }

            stream.Seek(chunkSize, SeekOrigin.Current);
        }

        throw new InvalidDataException("data chunk not found");
    }

    public static void Write(string path, float[] samples, int sampleRate, int channels)
    {
        using var stream = File.Create(path);
        WriteWav(stream, samples, sampleRate, channels);
    }

    private static void WriteWav(Stream stream, float[] samples, int sampleRate, int channels)
    {
        var byteRate = sampleRate * channels * 2;
        var dataSize = samples.Length * 2;
        WriteI32(stream, 0x46464952);
        WriteI32(stream, 36 + dataSize);
        WriteI32(stream, 0x45564157);
        WriteI32(stream, 0x20746D66);
        WriteI32(stream, 16);
        WriteI16(stream, 1);
        WriteI16(stream, (short)channels);
        WriteI32(stream, sampleRate);

        WriteI32(stream, byteRate);
        WriteI16(stream, (short)(channels * 2));
        WriteI16(stream, 16);
        WriteI32(stream, 0x61746164);
        WriteI32(stream, dataSize);

        foreach (var sample in samples)
        {
            var clamped = Math.Clamp(sample, -1f, 1f);
            WriteI16(stream, (short)(clamped * 32767f));
        }
    }

    private static void WriteI32(Stream stream, int value)
    {
        var bytes = BitConverter.GetBytes(value);
        stream.Write(bytes, 0, 4);
    }

    private static void WriteI16(Stream stream, short value)
    {
        var bytes = BitConverter.GetBytes(value);
        stream.Write(bytes, 0, 2);
    }
}
