namespace MossTtsSharp.Models;

internal class StreamingDecoder : IDisposable
{
    private readonly CodecDecoderStreaming _decoder;

    public StreamingDecoder(string codecModelDir)
    {
        _decoder = new CodecDecoderStreaming(codecModelDir);
    }

    public float[] DecodeFrame(int[] tokens) => _decoder.DecodeFrame(tokens);

    public float[] Flush() => _decoder.Flush();

    public void ResetCache() => _decoder.ResetCache();

    public void Dispose() => _decoder.Dispose();
}
