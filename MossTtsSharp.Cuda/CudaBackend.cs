using Microsoft.ML.OnnxRuntime;
using MossTtsSharp.Backend;

namespace MossTtsSharp.Cuda;

/// <summary>
/// CUDA backend for NVIDIA GPU acceleration.
/// </summary>
public class CudaBackend : BackendProvider
{
    public override SessionOptions CreateSessionOptions()
    {
        var opts = new SessionOptions();
        try
        {
            opts.AppendExecutionProvider_CUDA(0);
        }
        catch
        {
        }
        opts.AppendExecutionProvider_CPU();
        return opts;
    }
}