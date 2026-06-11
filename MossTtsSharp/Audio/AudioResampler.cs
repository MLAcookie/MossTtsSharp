namespace MossTtsSharp.Audio;

/// <summary>
/// Minimal audio resampler using linear interpolation.
/// </summary>
public static class AudioResampler
{
    /// <summary>
    /// Resample audio from srcSr to dstSr.
    /// samples interleaved: [channels * length]
    /// </summary>
    public static float[] Resample(float[] samples, int srcSr, int channels, int dstSr)
    {
        var ratio = (double)dstSr / srcSr;
        var srcLen = samples.Length / channels;
        var dstLen = (int)Math.Round(srcLen * ratio);

        var output = new float[dstLen * channels];

        for (var c = 0; c < channels; c++)
        {
            for (var i = 0; i < dstLen; i++)
            {
                var srcIndex = i / ratio;
                var idx0 = (int)Math.Floor(srcIndex);
                var idx1 = Math.Min(idx0 + 1, srcLen - 1);
                var frac = srcIndex - idx0;

                var v0 = samples[idx0 * channels + c];
                var v1 = samples[idx1 * channels + c];
                output[i * channels + c] = (float)(v0 * (1 - frac) + v1 * frac);
            }
        }

        return output;
    }
}
