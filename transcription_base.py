"""
Abstract base class for transcription backends.
Defines the common interface for Whisper, Qwen3-ASR, and future ASR models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, AsyncGenerator
from pathlib import Path


class TranscriptionBackend(ABC):
    """
    Abstract base class for all transcription backends.

    Each backend (Whisper, Qwen3-ASR, etc.) must implement these methods
    to provide a consistent interface for the application.
    """

    @abstractmethod
    def __init__(self):
        """Initialize the transcription backend."""
        pass

    @abstractmethod
    def get_available_models(self) -> Dict:
        """
        Get list of available models for this backend with their status.

        Returns:
            Dict mapping model_name -> {ram, installed, loaded}
            Example: {"tiny": {"ram": "4GB", "installed": True, "loaded": False}}
        """
        pass

    @abstractmethod
    async def load_model(self, model_name: str, websocket=None):
        """
        Load a transcription model.

        Args:
            model_name: Name of the model to load (e.g., "base", "qwen-1.7b")
            websocket: Optional WebSocket for progress updates

        Raises:
            Exception if model loading fails
        """
        pass

    @abstractmethod
    def unload(self):
        """
        Unload current model from memory and free resources.
        Should call torch.cuda.empty_cache() if using GPU.
        """
        pass

    @abstractmethod
    async def transcribe_file_streaming(
        self,
        file_path: str,
        language: Optional[str]
    ) -> AsyncGenerator[Dict, None]:
        """
        Transcribe an audio file with streaming updates.

        Args:
            file_path: Path to audio file
            language: Language code (e.g., "en", "es") or None for auto-detect

        Yields:
            Dict with keys:
                - type: "status", "partial_segment", "complete", "warning", "error"
                - For partial_segment: {"type": "partial_segment", "segment": {...}, "progress": int}
                - For complete: {"type": "complete", "segments": [...], "full_text": str}

        Segment format:
            {
                "text": str,
                "start": float (seconds),
                "end": float (seconds),
                "speaker": str (e.g., "Unknown", "Speaker 1")
            }
        """
        pass

    @abstractmethod
    async def transcribe_live(self, websocket, chunk_interval: int = 1):
        """
        Perform live transcription from browser audio stream via WebSocket.

        Args:
            websocket: FastAPI WebSocket connection
            chunk_interval: Number of 3-second chunks to accumulate before processing
                           (1 = 3 seconds, 3 = 9 seconds, 5 = 15 seconds)

        WebSocket Message Protocol:
            Client -> Server:
                - {"type": "init", "model": str, "language": str, "enableDiarization": bool, "chunkInterval": int}
                - {"type": "audio", "data": [int16], "sampleRate": int, "channels": int}
                - {"type": "stop", "enableDiarization": bool}

            Server -> Client:
                - {"type": "status", "message": str}
                - {"type": "transcription", "text": str, "start": float, "end": float, "speaker": str}
                - {"type": "diarization_complete", "segments": [...]}
                - {"type": "error", "message": str}
        """
        pass


class DiarizationMixin:
    """
    Mixin providing shared speaker diarization functionality.

    Both Whisper and Qwen3-ASR backends can use this to add speaker labels
    to transcribed segments.
    """

    def __init__(self):
        """Initialize diarization pipeline to None."""
        self.diarization_pipeline = None

    def _load_diarization_pipeline(self):
        """
        Load pyannote.audio speaker diarization pipeline.

        Requires:
            - pyannote.audio package installed
            - HF_TOKEN environment variable set
            - User acceptance of pyannote model terms

        Returns:
            Pipeline object or None if unavailable
        """
        import os
        import torch

        try:
            from pyannote.audio import Pipeline
        except ImportError:
            print("Warning: pyannote.audio not available. Speaker diarization will be disabled.")
            return None

        if self.diarization_pipeline is None:
            try:
                hf_token = os.getenv("HF_TOKEN")
                if hf_token:
                    # Login to HuggingFace hub
                    try:
                        from huggingface_hub import login
                        login(token=hf_token, add_to_git_credential=False)
                        print("Successfully logged in to HuggingFace Hub")
                    except Exception as e:
                        print(f"Warning: Could not login to HuggingFace Hub: {e}")

                    # Load pipeline
                    try:
                        print("Loading speaker diarization pipeline (this may take a moment)...")
                        self.diarization_pipeline = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1"
                        )

                        # Move to GPU if available
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        if device == "cuda":
                            self.diarization_pipeline.to(torch.device("cuda"))

                        print("Successfully loaded speaker diarization pipeline")
                    except Exception as e:
                        error_msg = str(e)
                        print(f"Warning: Could not load diarization pipeline: {error_msg}")

                        # Check if it's a permissions/gated model issue
                        if any(keyword in error_msg.lower() for keyword in ["gated", "private", "authenticate"]):
                            print("\nYou need to accept the terms for these HuggingFace models:")
                            print("1. https://huggingface.co/pyannote/speaker-diarization-3.1")
                            print("2. https://huggingface.co/pyannote/segmentation-3.0")
                            print("\nVisit each URL, click 'Agree and access repository', then restart the app.")

                        print("Speaker diarization will be disabled.")
                        self.diarization_pipeline = None
                else:
                    print("Warning: HF_TOKEN not set. Speaker diarization will be disabled.")
            except Exception as e:
                print(f"Warning: Could not load diarization pipeline: {e}")
                self.diarization_pipeline = None

        return self.diarization_pipeline

    def _apply_speaker_labels(self, diarization, segments):
        """
        Apply speaker labels to transcription segments.

        Args:
            diarization: pyannote.audio diarization result
            segments: List of segment dicts with start, end, text

        Returns:
            List of segments with updated speaker labels
        """
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            mid_time = (start_time + end_time) / 2

            # Find speaker at segment midpoint
            speaker_found = False
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start <= mid_time <= turn.end:
                    segment["speaker"] = f"Speaker {speaker}"
                    speaker_found = True
                    break

            # If no speaker found at midpoint, try overlap detection
            if not speaker_found:
                max_overlap = 0
                best_speaker = None
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    overlap_start = max(turn.start, start_time)
                    overlap_end = min(turn.end, end_time)
                    overlap = max(0, overlap_end - overlap_start)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_speaker = speaker

                if best_speaker:
                    segment["speaker"] = f"Speaker {best_speaker}"

        return segments

    def _merge_consecutive_segments(self, segments):
        """
        Merge consecutive segments from the same speaker.

        Args:
            segments: List of segment dicts with start, end, text, speaker

        Returns:
            List of merged segments
        """
        merged_segments = []
        for segment in segments:
            if merged_segments and merged_segments[-1]["speaker"] == segment["speaker"]:
                # Same speaker - merge with previous segment
                merged_segments[-1]["end"] = segment["end"]
                merged_segments[-1]["text"] += " " + segment["text"]
            else:
                # New speaker - add new segment
                merged_segments.append(segment.copy())

        return merged_segments

    def unload_diarization_pipeline(self):
        """Unload diarization pipeline from memory."""
        if self.diarization_pipeline is not None:
            del self.diarization_pipeline
            self.diarization_pipeline = None
