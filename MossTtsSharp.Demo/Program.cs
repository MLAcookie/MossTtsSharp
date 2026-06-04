using System.CommandLine;
using MossTtsSharp.Demo;

var textOption = new Option<string>("--text", "-t")
{
    Required = true,
    Description = "Text to synthesize"
};

var promptOption = new Option<FileInfo>("--prompt", "-p")
{
    Required = true,
    Description = "Audio prompt file (.wav, .mp3)"
};
promptOption.Validators.Add(result =>
{
    if (result.GetValueOrDefault<FileInfo>() is { Exists: false } file)
        result.AddError($"File '{file.FullName}' does not exist");
});

var outputOption = new Option<FileInfo>("--output", "-o")
{
    Description = "Output WAV file (if omitted, play directly)"
};

var modelsDirOption = new Option<DirectoryInfo>("--models-dir", "-m")
{
    Description = "Models root directory (default: ./OfficialOnnx)"
};

var noSampleOption = new Option<bool>("--no-sample")
{
    Description = "Disable random sampling"
};

var seedOption = new Option<int?>("--seed", "-s")
{
    Description = "Random seed for reproducibility"
};

var synCommand = new Command("syn", "Synthesize text to speech");
synCommand.Aliases.Add("synthesize");
synCommand.Options.Add(textOption);
synCommand.Options.Add(promptOption);
synCommand.Options.Add(outputOption);
synCommand.Options.Add(modelsDirOption);
synCommand.Options.Add(noSampleOption);
synCommand.Options.Add(seedOption);
synCommand.SetAction(ctx => TtsCommandHandler.ExecuteSynAsync(
    ctx.GetValue(textOption)!,
    ctx.GetValue(promptOption)!,
    ctx.GetValue(outputOption),
    ctx.GetValue(modelsDirOption),
    ctx.GetValue(noSampleOption),
    ctx.GetValue(seedOption)
));

var streamCommand = new Command("stream", "Stream text to speech");
streamCommand.Options.Add(textOption);
streamCommand.Options.Add(promptOption);
streamCommand.Options.Add(outputOption);
streamCommand.Options.Add(modelsDirOption);
streamCommand.Options.Add(noSampleOption);
streamCommand.Options.Add(seedOption);
streamCommand.SetAction(ctx => TtsCommandHandler.ExecuteStreamAsync(
    ctx.GetValue(textOption)!,
    ctx.GetValue(promptOption)!,
    ctx.GetValue(outputOption),
    ctx.GetValue(modelsDirOption),
    ctx.GetValue(noSampleOption),
    ctx.GetValue(seedOption)
));

var rootCommand = new RootCommand("MossTTS Demo");
rootCommand.Subcommands.Add(synCommand);
rootCommand.Subcommands.Add(streamCommand);

return await rootCommand.Parse(args).InvokeAsync();
