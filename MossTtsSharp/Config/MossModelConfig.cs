namespace MossTtsSharp.Config;

public static class MossModelConfig
{
    // ---- Model dimensions ----
    public const int HiddenSize = 768;
    public const int NumLayers = 12;
    public const int NumHeads = 12;
    public const int HeadDim = 64;
    public const int TextVocabSize = 16384;
    public const int Nvq = 16;
    public const int CodebookSize = 1024;
    public const int AudioPadTokenId = 1024;

    // ---- Special tokens ----
    public const int ImStartTokenId = 4;
    public const int ImEndTokenId = 5;
    public const int AudioStartTokenId = 6;
    public const int AudioEndTokenId = 7;
    public const int AudioUserSlotTokenId = 8;
    public const int AudioAssistantSlotTokenId = 9;

    // ---- Local transformer ----
    public const int LocalPositions = Nvq + 1; // 17

    // ---- Audio codec ----
    public const int SampleRate = 48000;
    public const int DownsampleRate = 3840;
    public const int Channels = 2;
    // ---- Default generation params ----
    public const int MaxNewFrames = 300;
    public const float TextTemperature = 1.0f;
    public const float TextTopP = 1.0f;
    public const int TextTopK = 50;
    public const float AudioTemperature = 0.8f;
    public const float AudioTopP = 0.95f;
    public const int AudioTopK = 25;
    public const float AudioRepetitionPenalty = 1.2f;
    public const bool DoSample = true;
}
