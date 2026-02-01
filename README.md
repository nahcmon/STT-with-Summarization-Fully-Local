# Speech-to-Text with Summarization

A web application for audio transcription with multiple backends (Whisper & Qwen3-ASR), featuring speaker diarization, target speaker extraction, and LLM-powered summarization via LM Studio.

## Features

- **Audio Input Options**:
  - Upload audio files (any format supported by FFmpeg)
  - Live recording from microphone

- **Multiple Transcription Backends**:
  - **Faster-Whisper**: Multiple model sizes (tiny, base, small, medium, large-v2, large-v3) with CUDA acceleration
  - **Qwen3-ASR**: Advanced transcription with forced alignment (0.6B, 1.7B models)
  - Automatic model download with progress tracking
  - CUDA acceleration for fast transcription
  - Model memory management (unload models to free VRAM)

- **Speaker Diarization**:
  - Automatic detection of multiple speakers using pyannote.audio
  - Speaker-tagged transcription segments
  - Works with both Whisper and Qwen backends

- **Target Speaker Extraction (TSE)**:
  - Enroll a speaker's voice from an audio sample
  - Extract and transcribe only that speaker's speech from multi-speaker recordings
  - Filter out background speakers and noise

- **LLM Integration**:
  - Connect to LM Studio for text summarization and analysis
  - Customizable system prompts
  - Model selection from LM Studio
  - Streaming responses

## Requirements

- Windows OS
- NVIDIA GPU with CUDA support (recommended)
- Python 3.9 or higher
- FFmpeg installed and in PATH
- CUDA Toolkit 12.1

## Quick Setup

### Option 1: Standard Setup (Whisper only)
```bash
setup.bat
```
This will:
- Create a Python virtual environment
- Install PyTorch with CUDA support
- Install all dependencies including Faster-Whisper
- Set up the application

### Option 2: Qwen3-ASR Setup (Advanced transcription)
```bash
setup_qwen.bat
```
This includes everything from Option 1 plus:
- Qwen3-ASR models for better transcription quality
- Optional flash-attention for improved performance

### Option 3: Speaker Diarization Setup
Speaker diarization and Target Speaker Extraction require a HuggingFace token.

```bash
setup_speaker_diarization.bat
```

This will guide you through:
1. Getting a HuggingFace token from https://huggingface.co/settings/tokens
2. Accepting terms for required models:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
3. Setting the HF_TOKEN environment variable

After setup, restart your terminal for the environment variable to take effect.

## Manual Installation

If you prefer manual setup:

### 1. Install CUDA Toolkit
Download and install CUDA Toolkit 12.1 from:
https://developer.nvidia.com/cuda-downloads

### 2. Install FFmpeg
Download FFmpeg from https://ffmpeg.org/download.html and add to PATH.

### 3. Install Python Dependencies
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### 4. Set HF_TOKEN (Optional - for speaker features)
```bash
setx HF_TOKEN "hf_your_token_here"
```

## Usage

### 1. Start the Application

**Choose one of these methods:**

**Method 1: Standard start (with speaker diarization if HF_TOKEN is set)**
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

**Choose a transcription backend:**
- **Whisper models** (whisper-tiny, whisper-base, whisper-small, whisper-medium, whisper-large-v2, whisper-large-v3)
- **Qwen3-ASR models** (qwen-0.6b, qwen-1.7b) - Better quality with forced alignment

**File Upload:**
1. Select a model from the dropdown
2. Choose "Upload File" tab
3. Select an audio file (any format supported by FFmpeg)
4. Click "Transcribe File"

**Live Recording:**
1. Select a model from the dropdown
2. Choose "Live Recording" tab
3. Click "Start Live Recording"
4. Speak into your microphone
5. Click "Stop Recording" when done

### 3. Target Speaker Extraction (TSE)

Extract only a specific speaker's voice from multi-speaker recordings:

1. **Enroll Speaker**: Upload a clean audio sample of the target speaker (3-10 seconds recommended)
2. **Extract Speaker**: Upload a multi-speaker audio file
3. The system will filter out other speakers and transcribe only the enrolled speaker

### 4. LLM Processing (Optional)

Process transcriptions with AI for summaries, analysis, or other text tasks:

1. Start LM Studio and load a model
2. In the web app, enter your LM Studio server URL (default: `http://localhost:1234`)
3. Click "Connect"
4. Select a model from the dropdown
5. Optionally customize the system prompt
6. Click "Send to LLM" to process your transcription
7. View streaming responses in real-time

## Model VRAM Requirements

### Whisper Models
- **whisper-tiny**: ~1GB VRAM
- **whisper-base**: ~1GB VRAM
- **whisper-small**: ~2GB VRAM
- **whisper-medium**: ~5GB VRAM
- **whisper-large-v2**: ~10GB VRAM
- **whisper-large-v3**: ~10GB VRAM

### Qwen3-ASR Models
- **qwen-0.6b**: ~2GB VRAM
- **qwen-1.7b**: ~5GB VRAM

### Speaker Diarization
- Adds ~2-3GB VRAM when enabled

## Memory Management

The web interface provides buttons to free up VRAM:
- **Unload Model**: Frees VRAM by unloading the current transcription model
- **Unload Model & Pipeline**: Frees VRAM by unloading both the transcription model and speaker diarization pipeline

## Troubleshooting

### Common Issues

**Installation error: "Could not build wheels for av"**
- The `requirements.txt` uses pre-built wheels (av==13.1.0)
- Delete the `venv` folder and run `setup.bat` again
- Ensure you have Visual C++ Build Tools installed

**CUDA not available**
- Update your NVIDIA GPU drivers
- Reinstall CUDA Toolkit 12.1 from https://developer.nvidia.com/cuda-downloads
- Reinstall PyTorch with CUDA support:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

**FFmpeg not found**
- Download FFmpeg from https://ffmpeg.org/download.html
- Extract and add the `bin` folder to your system PATH
- Restart your terminal after adding to PATH
- Verify with: `ffmpeg -version`

**Speaker diarization/TSE not working**
1. Get a HuggingFace token from https://huggingface.co/settings/tokens
2. Accept the terms for required models:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
3. Set the environment variable:
   ```bash
   setx HF_TOKEN "hf_your_token_here"
   ```
4. Restart your terminal/command prompt
5. Run `run.bat` or `run_with_token.bat`

**Qwen models not available**
- Run `setup_qwen.bat` to install qwen-asr package
- Or manually install: `pip install qwen-asr`
- For better performance, install flash-attention: `pip install flash-attn --no-build-isolation`

**Dependencies issues**
- Run `fix_dependencies.bat` to reinstall all packages
- Run `verify_installation.py` to check your setup
- Delete `venv` folder and run `setup.bat` for a fresh installation

## Project Structure

```
├── app.py                          # Main FastAPI application
├── transcription_handler.py        # Handles multiple transcription backends
├── transcription_base.py           # Base classes for transcription backends
├── whisper_transcriber.py          # Faster-Whisper implementation
├── whisper_handler.py              # Legacy Whisper handler
├── qwen_transcriber.py             # Qwen3-ASR implementation
├── tse_handler.py                  # Target Speaker Extraction
├── llm_handler.py                  # LM Studio integration
├── requirements.txt                # Python dependencies
├── config.example.py               # Configuration template
├── templates/
│   └── index.html                  # Web interface
├── static/
│   ├── style.css                   # Styles
│   └── script.js                   # Frontend JavaScript
└── Batch Scripts:
    ├── setup.bat                   # Standard setup (Whisper)
    ├── setup_qwen.bat              # Setup with Qwen3-ASR
    ├── setup_speaker_diarization.bat # Speaker features setup
    ├── run.bat                     # Run with diarization
    ├── run_no_diarization.bat      # Run without diarization
    ├── run_with_token.bat          # Run with token prompt
    ├── fix_dependencies.bat        # Reinstall dependencies
    ├── fix_diarization.bat         # Fix diarization issues
    ├── verify_installation.py      # Check installation
    └── check_syntax.py             # Syntax checker
```

## Configuration

You can customize the application by copying `config.example.py` to `config.py`:

```bash
copy config.example.py config.py
```

Available settings:
- Server host and port
- Default model selection
- Audio recording parameters
- LM Studio connection settings
- Speaker diarization options
- Upload limits and allowed file types
- Logging configuration

## Available Scripts

### Setup Scripts
- `setup.bat` - Standard installation with Faster-Whisper
- `setup_qwen.bat` - Installation with Qwen3-ASR support
- `setup_speaker_diarization.bat` - Interactive setup for speaker features

### Run Scripts
- `run.bat` - Standard run (enables diarization if HF_TOKEN is set)
- `run_no_diarization.bat` - Run without speaker diarization
- `run_with_token.bat` - Run with interactive token prompt

### Utility Scripts
- `verify_installation.py` - Verify all dependencies are correctly installed
- `check_syntax.py` - Check Python syntax in all project files
- `fix_dependencies.bat` - Reinstall all Python dependencies
- `fix_diarization.bat` - Fix speaker diarization issues

## API Endpoints

The application provides a REST API:

- `GET /` - Web interface
- `GET /api/transcription/models` - List available models from all backends
- `POST /api/transcription/unload` - Unload current model
- `POST /api/transcription/unload-all` - Unload all models and pipelines
- `POST /api/transcribe/file` - Transcribe uploaded file
- `POST /api/transcribe/file/stream` - Streaming transcription
- `WebSocket /api/transcribe/live` - Live transcription
- `POST /api/tse/enroll` - Enroll speaker for TSE
- `POST /api/tse/extract/stream` - Extract and transcribe target speaker
- `POST /api/lm-studio/connect` - Connect to LM Studio
- `GET /api/lm-studio/models` - Get available LLM models
- `POST /api/lm-studio/process` - Process text with LLM
- `POST /api/lm-studio/process/stream` - Streaming LLM processing

## Technologies Used

- **FastAPI** - Modern web framework
- **Faster-Whisper** - Optimized Whisper implementation
- **Qwen3-ASR** - Advanced ASR with forced alignment
- **PyAnnote.audio** - Speaker diarization and embeddings
- **PyTorch** - Deep learning framework with CUDA support
- **LM Studio** - Local LLM server integration

## License

MIT License
