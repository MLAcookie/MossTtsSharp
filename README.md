# MossTtsSharp

> **Warning**
>
> I am new to the AI field, and this project has not been extensively tested.
> **It is not recommended for production use.**
> If you have any suggestions or encounter issues, please open
> an [Issue](https://github.com/MLACookie/MossTtsSharp/issues).

An unofficial C# binding for [MOSS-TTS](https://github.com/OpenMOSS), a text-to-speech synthesis system powered by ONNX
Runtime.

## Project Structure

| Project                 | Description                                                            |
|-------------------------|------------------------------------------------------------------------|
| `MossTtsSharp`          | Core inference library — pipeline, tokenizer, audio codec, ONNX models |
| `MossTtsSharp.Backend`  | Backend abstraction for platform-specific ONNX Runtime                 |
| `MossTtsSharp.Cpu`      | CPU backend                                                            |
| `MossTtsSharp.Cuda`     | CUDA backend (NVIDIA GPU)                                              |
| `MossTtsSharp.DirectML` | DirectML backend (Windows GPU)                                         |
| `MossTtsSharp.Demo`     | CLI tool (`mosstts syn` / `mosstts stream`)                            |

## Quick Start

### Prerequisites

- [.NET SDK](https://dotnet.microsoft.com/download) 8.0 or later
- [Git](https://git-scm.com/) (for downloading models)

### 1. Download Models

```powershell
./DownloadModel.ps1
```

This clones the ONNX model repositories into `./MossTtsModel/`.

To specify a custom directory:

```powershell
./DownloadModel.ps1 -TargetDir "D:\Models\MossTTS"
```

To skip the overwrite confirmation:

```powershell
./DownloadModel.ps1 -Force
```

### 2. Run CLI Demo

```bash
dotnet run --project MossTtsSharp.Demo -- syn \
  -t "千里之行，始于足下。" \
  -p ./path/to/prompt.wav \
  -m ./MossTtsModel
```

**Streaming mode** (real-time playback):

```bash
dotnet run --project MossTtsSharp.Demo -- stream \
  -t "Hello, world!" \
  -p ./path/to/prompt.wav
```

### 3. Set Environment Variable (Optional)

Set `MOSSTTS_MODELS_DIR` to avoid passing `-m` every time:

```powershell
$env:MOSSTTS_MODELS_DIR = "D:\Models\MossTTS"
```

## CLI Reference

```
mosstts syn     -t <text> -p <audio> [-o <file>] [-m <dir>] [-n <value>]
mosstts stream  -t <text> -p <audio> [-o <file>] [-m <dir>] [-n <value>]
```

| Option             | Description                                                           |
|--------------------|-----------------------------------------------------------------------|
| `-t, --text`       | Text to synthesize **(required)**                                     |
| `-p, --prompt`     | Audio prompt file (.wav, .mp3) **(required)**                         |
| `-o, --output`     | Output WAV file (if omitted, play directly)                           |
| `-m, --models-dir` | Models root directory (or set `$MOSSTTS_MODELS_DIR`)                  |
| `-n, --noise`      | Fixed noise value for deterministic output (omit for random sampling) |

## API Usage

```csharp
using MossTtsSharp.Audio;
using MossTtsSharp.Config;
using MossTtsSharp.Cpu;
using MossTtsSharp.Pipeline;

// 1. Create configuration with CPU backend
var config = new MossConfig
{
    Device = new CpuBackend(),
    ModelsRoot = "./MossTtsModel"
};

// 2. Initialize pipeline
await using var pipeline = await MossTtsPipeline.CreateAsync(config);

// 3. Synthesize (batch)
var (waveform, sampleRate) = pipeline.Synthesize("你好，世界！", "prompt.wav");
AudioFile.Write("output.wav", waveform, sampleRate, MossModelConfig.Channels);

// 4. Or stream chunks in real-time
pipeline.SynthesizeStream("Hello, world!", "prompt.wav", audio =>
{
    // audio is float[] — feed to your audio device
});
```

### Choosing a Backend

```csharp
// CPU (default, works everywhere)
Device = new CpuBackend()

// NVIDIA GPU (Windows / Linux)
Device = new CudaBackend()

// Windows GPU via DirectML
Device = new DirectMlBackend()
```

Install the corresponding NuGet package alongside `MossTtsSharp`:

```
dotnet add package MossTtsSharp.Cpu       # CPU
dotnet add package MossTtsSharp.Cuda      # CUDA
dotnet add package MossTtsSharp.DirectML  # DirectML
```

## License

This project is licensed under the [Apache-2.0 License](LICENSE).

The ONNX models are from [MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) and are also licensed under Apache-2.0.
