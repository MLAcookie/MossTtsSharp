using Microsoft.ML.OnnxRuntime;
using MossTtsSharp.Backend;

namespace MossTtsSharp.Cpu;

/// <summary>
/// CPU-only backend. Uses the CPU execution provider.
/// </summary>
public class CpuBackend : BackendProvider
{
    public override SessionOptions CreateSessionOptions()
    {
        var opts = new SessionOptions();
        opts.AppendExecutionProvider_CPU();
        return opts;
    }
}