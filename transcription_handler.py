"""
Unified transcription handler managing Whisper and Qwen3-ASR backends.
"""

import torch
from typing import Dict, Optional

from transcription_base import TranscriptionBackend
from whisper_transcriber import WhisperTranscriber
from qwen_transcriber import QwenTranscriber, QWEN_AVAILABLE


class TranscriptionHandler:
    """
    Unified handler managing both Whisper and Qwen3-ASR backends.

    Routes model selection to the appropriate backend and provides
    a consistent API for file and live transcription.
    """

    def __init__(self):
        """Initialize both Whisper and Qwen3-ASR backends."""
        self.whisper = WhisperTranscriber()

        # Only initialize Qwen if available
        if QWEN_AVAILABLE:
            try:
                self.qwen = QwenTranscriber()
            except RuntimeError as e:
                print(f"Qwen3-ASR initialization failed: {e}")
                self.qwen = None
        else:
            self.qwen = None

        self.current_backend: Optional[TranscriptionBackend] = None
        self.current_model_name: Optional[str] = None

    def _get_backend(self, model_name: str) -> TranscriptionBackend:
        """
        Route model name to appropriate backend.

        Args:
            model_name: Model name (e.g., "whisper-base", "qwen-1.7b", "base")

        Returns:
            TranscriptionBackend instance (Whisper or Qwen)

        Raises:
            ValueError: If Qwen models requested but not available
        """
        if model_name.startswith("qwen-"):
            if not self.qwen:
                raise ValueError(
                    "Qwen3-ASR models not available. Install with: pip install qwen-asr"
                )
            return self.qwen
        else:
            # Whisper models (including legacy names without "whisper-" prefix)
            return self.whisper

    def get_available_models(self) -> Dict:
        """
        Get all available models from both backends.

        Returns:
            Dict with keys "whisper" and "qwen", each containing model info
        """
        models = {
            "whisper": self.whisper.get_available_models(),
            "qwen": self.qwen.get_available_models() if self.qwen else {}
        }
        return models

    async def load_model(self, model_name: str, websocket=None):
        """
        Load a model from the appropriate backend.

        Args:
            model_name: Model name to load
            websocket: Optional WebSocket for progress updates
        """
        backend = self._get_backend(model_name)

        # Switch backends if needed
        if self.current_backend != backend:
            # Unload previous backend
            if self.current_backend:
                self.current_backend.unload()
                self.current_backend = None

        # Load model in selected backend
        await backend.load_model(model_name, websocket)

        self.current_backend = backend
        self.current_model_name = model_name

    def unload_model(self):
        """Unload current transcription model."""
        if self.current_backend:
            self.current_backend.unload()
            self.current_backend = None
            self.current_model_name = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_all(self):
        """Unload all models and diarization pipelines from both backends."""
        # Unload Whisper
        self.whisper.unload()
        if hasattr(self.whisper, 'diarization_pipeline'):
            self.whisper.unload_diarization_pipeline()

        # Unload Qwen
        if self.qwen:
            self.qwen.unload()
            if hasattr(self.qwen, 'diarization_pipeline'):
                self.qwen.unload_diarization_pipeline()

        self.current_backend = None
        self.current_model_name = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def transcribe_file_streaming(
        self,
        file_path: str,
        model_name: str,
        language: Optional[str] = None
    ):
        """
        Transcribe an audio file with streaming updates.

        Args:
            file_path: Path to audio file
            model_name: Model to use for transcription
            language: Language code or None for auto-detect

        Yields:
            Dict with transcription updates
        """
        # Get appropriate backend
        backend = self._get_backend(model_name)

        # Load model if not already loaded
        if self.current_backend != backend or self.current_model_name != model_name:
            await self.load_model(model_name)

        # Stream transcription results from backend
        async for update in backend.transcribe_file_streaming(file_path, language):
            yield update

    async def transcribe_live(
        self,
        websocket,
        model_name: str = "whisper-base",
        chunk_interval: int = 1
    ):
        """
        Perform live transcription from browser audio stream.

        Args:
            websocket: FastAPI WebSocket connection
            model_name: Model to use (from init message)
            chunk_interval: Number of 3s chunks to accumulate (for Qwen)

        Note:
            - Whisper processes each chunk immediately (chunk_interval ignored)
            - Qwen accumulates chunks based on chunk_interval setting
        """
        # Get appropriate backend
        backend = self._get_backend(model_name)

        # Set current backend (model loading happens in backend's transcribe_live)
        self.current_backend = backend

        # Delegate to backend's live transcription
        await backend.transcribe_live(websocket, chunk_interval)

    def get_current_model_info(self) -> Dict:
        """
        Get information about currently loaded model.

        Returns:
            Dict with model_name, backend, and loaded status
        """
        return {
            "model_name": self.current_model_name,
            "backend": "whisper" if self.current_backend == self.whisper else "qwen" if self.current_backend == self.qwen else None,
            "loaded": self.current_backend is not None
        }
