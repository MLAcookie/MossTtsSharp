using Microsoft.ML.OnnxRuntime;

namespace MossTtsSharp.Backend;

/// <summary>
/// Abstract base for platform-specific ONNX Runtime session configuration.
/// Implementations create <see cref="SessionOptions"/> with the appropriate
/// execution provider (DML, CUDA, CPU, etc.).
/// </summary>
public abstract class BackendProvider
{
    /// <summary>Configure and return <see cref="SessionOptions"/> for this backend.</summary>
    public abstract SessionOptions CreateSessionOptions();
}
