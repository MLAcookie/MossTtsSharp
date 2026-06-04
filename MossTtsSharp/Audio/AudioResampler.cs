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
        double ratio = (double)dstSr / srcSr;
        int srcLen = samples.Length / channels;
        int dstLen = (int)Math.Round(srcLen * ratio);

        float[] output = new float[dstLen * channels];

        for (int c = 0; c < channels; c++)
        {
            for (int i = 0; i < dstLen; i++)
            {
                double srcIndex = i / ratio;
                int idx0 = (int)Math.Floor(srcIndex);
                int idx1 = Math.Min(idx0 + 1, srcLen - 1);
                double frac = srcIndex - idx0;

                float v0 = samples[idx0 * channels + c];
                float v1 = samples[idx1 * channels + c];
                output[i * channels + c] = (float)(v0 * (1 - frac) + v1 * frac);
            }
        }

        return output;
    }
}
