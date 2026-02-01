# Speech-to-Text with Summarization

A web application for audio transcription using OpenAI Whisper with CUDA acceleration, featuring speaker diarization and LLM-powered summarization via LM Studio.

## Features

- **Audio Input Options**:
  - Upload audio files (any format supported by FFmpeg)
  - Live recording from microphone

- **Whisper Models**:
  - Multiple model sizes (tiny, base, small, medium, large-v2, large-v3)
  - Automatic model download with progress tracking
  - CUDA acceleration for fast transcription
  - Model memory management (unload models to free VRAM)

- **Speaker Diarization**:
  - Automatic detection of multiple speakers
  - Speaker-tagged transcription segments

- **LLM Integration**:
  - Connect to LM Studio for text summarization
  - Customizable system prompts
  - Model selection from LM Studio

## Requirements

- Windows OS
- NVIDIA GPU with CUDA support
- Python 3.9 or higher
- FFmpeg installed and in PATH
- CUDA Toolkit 12.1

## Installation

### 1. Install CUDA Toolkit

Download and install CUDA Toolkit 12.1 from:
https://developer.nvidia.com/cuda-downloads

### 2. Install FFmpeg

Download FFmpeg from https://ffmpeg.org/download.html and add to PATH.

### 3. Install Python Dependencies

```bash
# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### 4. Set up Speaker Diarization (OPTIONAL)

Speaker diarization detects and labels different speakers. **You can skip this if you don't need speaker identification.**

**Option A: Skip speaker diarization** (easiest)
```bash
run_no_diarization.bat
```
All speakers will be labeled as "Unknown"

**Option B: Automated setup** (recommended if you want speaker diarization)
```bash
setup_speaker_diarization.bat
```
This will guide you through:
- Getting a HuggingFace token
- Accepting terms for required models
- Setting the environment variable

**Option C: Manual setup**

See detailed instructions in [SPEAKER_DIARIZATION_SETUP.md](SPEAKER_DIARIZATION_SETUP.md)

**Quick summary:**
1. Get token: https://huggingface.co/settings/tokens
2. Accept terms for these models:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/segmentation
3. Set HF_TOKEN: `setx HF_TOKEN "hf_your_token"`
4. Open NEW Command Prompt and run `run.bat`

## Usage

### 1. Start the Application

**Choose one of these methods:**

**Method 1: Simple start (with speaker diarization if HF_TOKEN is set)**
```bash
run.bat
```

**Method 2: Without speaker diarization (easiest, no token needed)**
```bash
run_no_diarization.bat
```

**Method 3: Interactive token prompt**
```bash
run_with_token.bat
```

**Method 4: Manual start**
```bash
venv\Scripts\activate
python app.py
```

The application will start on `http://localhost:3456`

Open your browser and navigate to: **http://localhost:3456**

### 2. Transcribe Audio

**File Upload:**
1. Select a Whisper model from the dropdown
2. Choose "Upload File" tab
3. Select an audio file
4. Click "Transcribe File"

**Live Recording:**
1. Select a Whisper model
2. Choose "Live Recording" tab
3. Click "Start Live Recording"
4. Speak into your microphone
5. Click "Stop Recording" when done

### 3. LLM Processing (Optional)

1. Start LM Studio and load a model
2. In the web app, enter your LM Studio server URL (default: `http://localhost:1234`)
3. Click "Connect"
4. Select a model from the dropdown
5. Optionally customize the system prompt
6. Click "Send to LLM" to process your transcription

## Model RAM Requirements

- **tiny**: 4GB VRAM
- **base**: 5GB VRAM
- **small**: 6GB VRAM
- **medium**: 10GB VRAM
- **large-v2**: 14GB VRAM
- **large-v3**: 14+GB VRAM

## Memory Management

- **Unload Model**: Frees VRAM by unloading the Whisper model
- **Unload Model & Pipeline**: Frees VRAM by unloading both Whisper and the diarization pipeline

## Troubleshooting

For detailed troubleshooting steps, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### Common Issues

**Installation error: "Could not build wheels for av"**
- Updated `requirements.txt` now uses pre-built wheels
- Delete `venv` folder and run `setup.bat` again
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#error-could-not-build-wheels-for-av) for details

**CUDA not available**
- Update NVIDIA drivers
- Reinstall CUDA Toolkit 12.1
- Reinstall PyTorch with CUDA support
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#error-cuda-not-available-or-torchcudais_available-returns-false)

**FFmpeg not found**
- Download FFmpeg and add to PATH
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#error-ffmpeg-not-found)

**Speaker diarization not working**
- Set HF_TOKEN environment variable
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#error-huggingface-authentication-token-required)

**For all other issues**, consult the [TROUBLESHOOTING.md](TROUBLESHOOTING.md) guide.

## License

MIT License
