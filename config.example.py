"""
Configuration file for the Speech-to-Text application
Copy this file to config.py and modify as needed
"""

# Server configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

# Whisper model settings
DEFAULT_WHISPER_MODEL = "base"
WHISPER_DEVICE = "cuda"  # "cuda" or "cpu"
WHISPER_COMPUTE_TYPE = "float16"  # "float16" for CUDA, "int8" for CPU

# Audio recording settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 5  # seconds
TRANSCRIPTION_INTERVAL = 3  # seconds between live transcriptions

# LM Studio settings
DEFAULT_LM_STUDIO_URL = "http://localhost:1234"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = -1  # -1 for unlimited

# Speaker diarization settings
ENABLE_DIARIZATION = True
DIARIZATION_MIN_SPEAKERS = 1
DIARIZATION_MAX_SPEAKERS = 10

# File upload settings
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_AUDIO_EXTENSIONS = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"]

# Paths
UPLOAD_DIR = "uploads"
TEMP_DIR = "temp"
MODEL_CACHE_DIR = None  # None to use default HuggingFace cache

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "app.log"
ENABLE_FILE_LOGGING = False
