using Microsoft.ML.OnnxRuntime;
using MossTtsSharp.Backend;

namespace MossTtsSharp.DirectML;

/// <summary>
/// DirectML backend for Windows GPU acceleration using DirectX 12.
/// </summary>
public class DirectMlBackend : BackendProvider
{
    public override SessionOptions CreateSessionOptions()
    {
        var opts = new SessionOptions();
        try
        {
            opts.AppendExecutionProvider_DML(0);
        }
        catch
        {
        }
        opts.AppendExecutionProvider_CPU();
        return opts;
    }
}
