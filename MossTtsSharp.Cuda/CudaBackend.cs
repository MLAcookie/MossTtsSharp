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
        opts.AppendExecutionProvider_CUDA(0);
        return opts;
    }
}
