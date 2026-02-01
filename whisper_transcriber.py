"""
Whisper transcription backend using faster-whisper.
"""

import os
import asyncio
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import soundfile as sf
from faster_whisper import WhisperModel
import time

from transcription_base import TranscriptionBackend, DiarizationMixin


class WhisperTranscriber(TranscriptionBackend, DiarizationMixin):
    """
    Whisper transcription backend using faster-whisper library.
    Supports CUDA acceleration and speaker diarization.
    """

    MODELS = {
        "whisper-tiny": {"ram": "4GB", "size": "tiny"},
        "whisper-base": {"ram": "5GB", "size": "base"},
        "whisper-small": {"ram": "6GB", "size": "small"},
        "whisper-medium": {"ram": "10GB", "size": "medium"},
        "whisper-large-v2": {"ram": "14GB", "size": "large-v2"},
        "whisper-large-v3": {"ram": "14+GB", "size": "large-v3"},
        # Legacy support (without "whisper-" prefix)
        "tiny": {"ram": "4GB", "size": "tiny"},
        "base": {"ram": "5GB", "size": "base"},
        "small": {"ram": "6GB", "size": "small"},
        "medium": {"ram": "10GB", "size": "medium"},
        "large-v2": {"ram": "14GB", "size": "large-v2"},
        "large-v3": {"ram": "14+GB", "size": "large-v3"},
    }

    def __init__(self):
        """Initialize Whisper transcriber."""
        TranscriptionBackend.__init__(self)
        DiarizationMixin.__init__(self)

        self.model: Optional[WhisperModel] = None
        self.current_model_name: Optional[str] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.model_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    def _normalize_model_name(self, model_name: str) -> str:
        """
        Normalize model name to support both "base" and "whisper-base" formats.

        Args:
            model_name: Original model name

        Returns:
            Tuple of (normalized_name_with_prefix, whisper_model_size)
        """
        # Strip "whisper-" prefix if present to get the model size
        if model_name.startswith("whisper-"):
            size = model_name[8:]  # Remove "whisper-" prefix
            normalized = model_name
        else:
            size = model_name
            normalized = f"whisper-{model_name}"

        return normalized, size

    def get_available_models(self) -> Dict:
        """Get list of available Whisper models with their status."""
        models = {}
        # Only return models with "whisper-" prefix to avoid duplicates
        for name, info in self.MODELS.items():
            if name.startswith("whisper-"):
                models[name] = {
                    "ram": info["ram"],
                    "installed": self._is_model_installed(info["size"]),
                    "loaded": self.current_model_name == name
                }
        return models

    def _is_model_installed(self, model_size: str) -> bool:
        """Check if a Whisper model is already downloaded."""
        model_path = self.model_cache_dir / f"models--Systran--faster-whisper-{model_size}"
        return model_path.exists()

    async def _download_model_with_progress(self, model_size: str, websocket=None):
        """Download Whisper model with progress reporting."""
        try:
            if websocket:
                await websocket.send_json({
                    "type": "download_progress",
                    "status": "starting",
                    "message": f"Starting download of Whisper {model_size} model..."
                })

            start_time = time.time()

            def load_model():
                return WhisperModel(
                    model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=str(self.model_cache_dir)
                )

            # Run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(None, load_model)

            elapsed = time.time() - start_time

            if websocket:
                await websocket.send_json({
                    "type": "download_progress",
                    "status": "complete",
                    "message": f"Whisper model loaded in {elapsed:.2f} seconds"
                })

            return True

        except Exception as e:
            if websocket:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Failed to load Whisper model: {str(e)}"
                })
            raise

    async def load_model(self, model_name: str, websocket=None):
        """Load a Whisper model."""
        normalized_name, model_size = self._normalize_model_name(model_name)

        if self.current_model_name == normalized_name and self.model is not None:
            return  # Model already loaded

        # Unload current model if any
        self.unload()

        # Download/load new model
        await self._download_model_with_progress(model_size, websocket)
        self.current_model_name = normalized_name

    def unload(self):
        """Unload Whisper model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.current_model_name = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def transcribe_file_streaming(self, file_path: str, language: Optional[str]):
        """Transcribe an audio file with streaming updates."""
        yield {"type": "status", "message": "Model loaded, starting transcription..."}

        # Start transcription in background
        loop = asyncio.get_event_loop()

        def transcribe_worker():
            """Worker function to run transcription in thread"""
            return self.model.transcribe(
                file_path,
                beam_size=5,
                language=language if language and language != 'auto' else None,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

        # Get segments generator and info
        segments_generator, info = await loop.run_in_executor(None, transcribe_worker)

        # Collect segments and send them as they arrive
        transcription_segments = []
        full_text = ""

        # Process each segment asynchronously
        def get_next_segment(gen):
            """Get next segment from generator"""
            try:
                return next(gen), False
            except StopIteration:
                return None, True

        while True:
            # Get next segment in executor to avoid blocking
            segment, done = await loop.run_in_executor(None, get_next_segment, segments_generator)

            if done:
                break

            seg_data = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "speaker": "Unknown"
            }
            transcription_segments.append(seg_data)
            full_text += segment.text + " "

            # Send partial result immediately
            yield {
                "type": "partial_segment",
                "segment": seg_data,
                "progress": len(transcription_segments)
            }

        yield {"type": "status", "message": "Transcription complete, running speaker diarization..."}

        # Apply speaker diarization if available
        pipeline = self._load_diarization_pipeline()
        if pipeline:
            try:
                diarization = await loop.run_in_executor(None, lambda: pipeline(file_path))
                transcription_segments = self._apply_speaker_labels(diarization, transcription_segments)
            except Exception as e:
                print(f"Diarization failed: {e}")
                yield {"type": "warning", "message": f"Speaker diarization failed: {str(e)}"}

        # Merge consecutive segments from the same speaker
        merged_segments = self._merge_consecutive_segments(transcription_segments)

        # Update full text from merged segments
        full_text = " ".join(seg["text"] for seg in merged_segments)

        # Send final result
        yield {
            "type": "complete",
            "language": info.language if hasattr(info, 'language') else "unknown",
            "segments": merged_segments,
            "full_text": full_text.strip()
        }

    async def transcribe_live(self, websocket, chunk_interval: int = 1):
        """
        Perform live transcription from browser audio stream.

        Note: Whisper processes each 3-second chunk independently, so chunk_interval
        is always 1 for Whisper (immediate processing). The parameter is included
        for API consistency with other backends.
        """
        audio_chunks = []
        chunk_count = 0
        loop = asyncio.get_event_loop()
        model_loaded = False
        language = None
        enable_diarization = True

        # Storage for full recording
        all_audio_data = []
        all_segments = []
        total_duration = 0.0

        # Helper function to safely send messages
        async def safe_send(message):
            try:
                await websocket.send_json(message)
                return True
            except Exception as e:
                print(f"Could not send message (WebSocket closed): {e}")
                return False

        try:
            while True:
                # Receive message from browser
                try:
                    message = await websocket.receive_json()

                    if message.get("type") == "stop":
                        enable_diarization = message.get("enableDiarization", True)
                        break

                    elif message.get("type") == "init":
                        # Initial connection with model info
                        model_name = message.get('model', 'base')
                        language = message.get('language', 'auto')
                        enable_diarization = message.get('enableDiarization', True)
                        # chunk_interval is ignored for Whisper (always processes immediately)
                        print(f"Whisper live recording initialized with model: {model_name}, language: {language}")

                        # Load model (with progress updates via websocket)
                        await self.load_model(model_name, websocket)
                        model_loaded = True

                        await websocket.send_json({
                            "type": "status",
                            "message": "Server ready - start speaking!"
                        })

                    elif message.get("type") == "audio":
                        # Received audio chunk from browser
                        if not model_loaded:
                            print("Warning: Received audio before model was loaded")
                            continue

                        audio_data = message.get("data")
                        sample_rate = message.get("sampleRate", 16000)
                        channels = message.get("channels", 1)

                        if audio_data:
                            chunk_count += 1
                            print(f"Received audio chunk {chunk_count}, samples: {len(audio_data)}, rate: {sample_rate}Hz")

                            # Store audio data for later diarization
                            all_audio_data.extend(audio_data)

                            # Calculate chunk duration
                            chunk_duration = len(audio_data) / sample_rate

                            temp_file = None
                            try:
                                # Convert list of int16 values to numpy array
                                audio_array = np.array(audio_data, dtype=np.int16)

                                # Convert to float32 for soundfile
                                audio_float = audio_array.astype(np.float32) / 32768.0

                                # Save to temporary WAV file
                                temp_file = f"temp_live_audio_{int(time.time() * 1000)}.wav"

                                def save_wav():
                                    sf.write(temp_file, audio_float, sample_rate)

                                await loop.run_in_executor(None, save_wav)

                                # Send status update
                                await websocket.send_json({
                                    "type": "status",
                                    "message": f"Transcribing chunk {chunk_count}..."
                                })

                                # Transcribe in executor to avoid blocking
                                def transcribe_chunk():
                                    try:
                                        segments, info = self.model.transcribe(
                                            temp_file,
                                            beam_size=5,
                                            language=language if language and language != 'auto' else None,
                                            vad_filter=True
                                        )
                                        return list(segments), None
                                    except Exception as e:
                                        import traceback
                                        return None, str(e) + "\n" + traceback.format_exc()

                                segments_list, error = await loop.run_in_executor(None, transcribe_chunk)

                                if error:
                                    print(f"Transcription error: {error}")
                                    await websocket.send_json({
                                        "type": "status",
                                        "message": "Processing audio..."
                                    })
                                elif segments_list:
                                    # Send results and store for later diarization
                                    for segment in segments_list:
                                        if segment.text.strip():
                                            # Add offset to timestamps
                                            adjusted_start = total_duration + segment.start
                                            adjusted_end = total_duration + segment.end

                                            seg_data = {
                                                "text": segment.text,
                                                "start": adjusted_start,
                                                "end": adjusted_end,
                                                "speaker": "Unknown"
                                            }
                                            all_segments.append(seg_data)

                                            await websocket.send_json({
                                                "type": "transcription",
                                                "text": segment.text,
                                                "start": adjusted_start,
                                                "end": adjusted_end,
                                                "speaker": "Unknown"
                                            })
                                            print(f"Sent transcription: {segment.text} ({adjusted_start:.1f}s - {adjusted_end:.1f}s)")

                                    total_duration += chunk_duration
                                else:
                                    print("No speech detected in chunk")
                                    await websocket.send_json({
                                        "type": "status",
                                        "message": "No speech detected"
                                    })
                                    total_duration += chunk_duration

                            except Exception as e:
                                import traceback
                                error_msg = f"Live transcription error: {e}\n{traceback.format_exc()}"
                                print(error_msg)
                                await websocket.send_json({
                                    "type": "error",
                                    "message": f"Transcription error: {str(e)}"
                                })
                            finally:
                                # Clean up temp file
                                if temp_file and os.path.exists(temp_file):
                                    try:
                                        os.remove(temp_file)
                                    except Exception:
                                        pass

                except Exception as e:
                    print(f"WebSocket error: {e}")
                    break

        finally:
            # Run speaker diarization on complete recording if enabled
            print(f"Live recording ended. Diarization enabled: {enable_diarization}")

            if enable_diarization and all_audio_data and all_segments:
                try:
                    print("Starting speaker diarization...")
                    if not await safe_send({
                        "type": "status",
                        "message": "Running speaker diarization on complete recording..."
                    }):
                        print("WebSocket closed, skipping diarization updates")
                        return

                    # Convert accumulated audio to WAV
                    full_audio_array = np.array(all_audio_data, dtype=np.int16)
                    full_audio_float = full_audio_array.astype(np.float32) / 32768.0
                    full_audio_file = f"temp_live_complete_{int(time.time() * 1000)}.wav"

                    def save_full_wav():
                        sf.write(full_audio_file, full_audio_float, 16000)

                    await loop.run_in_executor(None, save_full_wav)

                    # Run diarization
                    pipeline = self._load_diarization_pipeline()
                    if pipeline:
                        def run_diarization():
                            return pipeline(full_audio_file)

                        diarization = await loop.run_in_executor(None, run_diarization)

                        # Apply speaker labels
                        all_segments = self._apply_speaker_labels(diarization, all_segments)

                        # Merge consecutive same-speaker segments
                        merged_segments = self._merge_consecutive_segments(all_segments)

                        # Send updated segments with speakers
                        await safe_send({
                            "type": "diarization_complete",
                            "segments": merged_segments
                        })

                        print(f"Diarization complete, {len(merged_segments)} segments with speakers")

                    # Clean up
                    try:
                        os.remove(full_audio_file)
                    except:
                        pass

                except Exception as e:
                    import traceback
                    print(f"Diarization error: {e}\n{traceback.format_exc()}")
                    await safe_send({
                        "type": "warning",
                        "message": "Speaker diarization failed"
                    })

            # Send final status
            await safe_send({
                "type": "status",
                "message": "Recording stopped"
            })
            print("Whisper live transcription session ended")
