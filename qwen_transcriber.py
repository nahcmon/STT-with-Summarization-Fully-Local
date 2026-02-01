"""
Qwen3-ASR transcription backend using qwen-asr library.
"""

import os
import asyncio
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import soundfile as sf
import time

from transcription_base import TranscriptionBackend, DiarizationMixin

# Try to import qwen-asr
try:
    from qwen_asr import Qwen3ASRModel
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    print("Warning: qwen-asr not installed. Qwen models will be unavailable.")
    print("Install with: pip install qwen-asr")


class QwenTranscriber(TranscriptionBackend, DiarizationMixin):
    """
    Qwen3-ASR transcription backend using qwen-asr library.
    Supports CUDA acceleration, forced alignment for timestamps, and speaker diarization.
    """

    MODELS = {
        "qwen-0.6b": {
            "hf_name": "Qwen/Qwen3-ASR-0.6B",
            "ram": "2GB",
            "aligner": "Qwen/Qwen3-ForcedAligner-0.6B"
        },
        "qwen-1.7b": {
            "hf_name": "Qwen/Qwen3-ASR-1.7B",
            "ram": "5GB",
            "aligner": "Qwen/Qwen3-ForcedAligner-0.6B"
        }
    }

    def __init__(self):
        """Initialize Qwen3-ASR transcriber."""
        if not QWEN_AVAILABLE:
            raise RuntimeError(
                "qwen-asr package not installed. Install with: pip install qwen-asr\n"
                "Or run: setup_qwen.bat"
            )

        TranscriptionBackend.__init__(self)
        DiarizationMixin.__init__(self)

        self.model: Optional[Qwen3ASRModel] = None
        self.current_model_name: Optional[str] = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    def get_available_models(self) -> Dict:
        """Get list of available Qwen3-ASR models with their status."""
        if not QWEN_AVAILABLE:
            return {}

        models = {}
        for name, info in self.MODELS.items():
            models[name] = {
                "ram": info["ram"],
                "installed": self._is_model_installed(info["hf_name"]),
                "loaded": self.current_model_name == name
            }
        return models

    def _is_model_installed(self, hf_model_name: str) -> bool:
        """Check if a Qwen3-ASR model is already downloaded."""
        # Convert "Qwen/Qwen3-ASR-1.7B" to "models--Qwen--Qwen3-ASR-1.7B"
        model_dir_name = "models--" + hf_model_name.replace("/", "--")
        model_path = self.model_cache_dir / model_dir_name
        return model_path.exists()

    async def _download_model_with_progress(self, model_name: str, websocket=None):
        """Download Qwen3-ASR model with progress reporting."""
        try:
            model_config = self.MODELS[model_name]

            if websocket:
                await websocket.send_json({
                    "type": "download_progress",
                    "status": "starting",
                    "message": f"Loading {model_name} model (this may take a while on first use)..."
                })

            start_time = time.time()

            def load_model():
                return Qwen3ASRModel.from_pretrained(
                    model_config["hf_name"],
                    dtype=self.dtype,
                    device_map=self.device,
                    forced_aligner=model_config["aligner"],
                    forced_aligner_kwargs=dict(
                        dtype=self.dtype,
                        device_map=self.device
                    ),
                    max_inference_batch_size=8,
                    max_new_tokens=256,
                )

            # Run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(None, load_model)

            elapsed = time.time() - start_time

            if websocket:
                await websocket.send_json({
                    "type": "download_progress",
                    "status": "complete",
                    "message": f"Qwen3-ASR model loaded in {elapsed:.2f} seconds"
                })

            return True

        except Exception as e:
            if websocket:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Failed to load Qwen3-ASR model: {str(e)}"
                })
            raise

    async def load_model(self, model_name: str, websocket=None):
        """Load a Qwen3-ASR model with ForcedAligner."""
        if self.current_model_name == model_name and self.model is not None:
            return  # Model already loaded

        # Unload current model if any
        self.unload()

        # Download/load new model
        await self._download_model_with_progress(model_name, websocket)
        self.current_model_name = model_name

    def unload(self):
        """Unload Qwen3-ASR model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.current_model_name = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def transcribe_file_streaming(self, file_path: str, language: Optional[str]):
        """
        Transcribe an audio file with progressive timestamp yielding.

        Note: Qwen3-ASR processes the entire file (not chunk-by-chunk like Whisper).
        We yield word/character-level timestamps progressively to simulate streaming.
        """
        yield {"type": "status", "message": "Processing with Qwen3-ASR..."}

        loop = asyncio.get_event_loop()

        def transcribe():
            """Run Qwen3-ASR transcription with ForcedAligner for timestamps."""
            return self.model.transcribe(
                audio=file_path,
                language=language if language and language != 'auto' else None,
                return_time_stamps=True  # Requires ForcedAligner
            )

        # Run transcription (processes entire file)
        results = await loop.run_in_executor(None, transcribe)

        # Yield timestamps as segments
        transcription_segments = []
        if results and len(results) > 0:
            result = results[0]

            # Check if timestamps are available
            if hasattr(result, 'time_stamps') and result.time_stamps:
                for ts in result.time_stamps:
                    segment = {
                        "text": ts.text,
                        "start": ts.start_time,
                        "end": ts.end_time,
                        "speaker": "Unknown"
                    }
                    transcription_segments.append(segment)

                    # Yield each timestamp as a partial segment
                    yield {
                        "type": "partial_segment",
                        "segment": segment,
                        "progress": len(transcription_segments)
                    }
            else:
                # Fallback if no timestamps (shouldn't happen with ForcedAligner)
                segment = {
                    "text": result.text,
                    "start": 0.0,
                    "end": 0.0,
                    "speaker": "Unknown"
                }
                transcription_segments.append(segment)
                yield {
                    "type": "partial_segment",
                    "segment": segment,
                    "progress": 1
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

        # Build full text
        full_text = " ".join(seg["text"] for seg in merged_segments)

        # Send final result
        yield {
            "type": "complete",
            "language": results[0].language if results and hasattr(results[0], 'language') else "unknown",
            "segments": merged_segments,
            "full_text": full_text.strip()
        }

    async def transcribe_live(self, websocket, chunk_interval: int = 3):
        """
        Perform live transcription from browser audio stream.

        Args:
            chunk_interval: Number of 3-second chunks to accumulate before processing
                           (1 = 3s, 3 = 9s, 5 = 15s)
                           Default: 3 (9 seconds) for balanced latency/accuracy
        """
        chunk_buffer = []
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
                        model_name = message.get('model', 'qwen-0.6b')
                        language = message.get('language', 'auto')
                        enable_diarization = message.get('enableDiarization', True)
                        chunk_interval = message.get('chunkInterval', 3)  # Use configurable interval
                        print(f"Qwen live recording initialized: model={model_name}, language={language}, chunk_interval={chunk_interval}")

                        # Load model (with progress updates via websocket)
                        await self.load_model(model_name, websocket)
                        model_loaded = True

                        await websocket.send_json({
                            "type": "status",
                            "message": f"Server ready - processing every {chunk_interval * 3} seconds"
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
                            # Add to buffer
                            chunk_buffer.extend(audio_data)
                            chunk_count += 1

                            # Store all audio for final diarization
                            all_audio_data.extend(audio_data)

                            print(f"Received chunk {chunk_count}, buffer size: {len(chunk_buffer)} samples")

                            # Process when we've accumulated enough chunks
                            if chunk_count >= chunk_interval:
                                temp_file = None
                                try:
                                    # Convert buffer to numpy array
                                    audio_array = np.array(chunk_buffer, dtype=np.int16)
                                    audio_float = audio_array.astype(np.float32) / 32768.0

                                    # Save to temporary WAV file
                                    temp_file = f"temp_qwen_live_{int(time.time() * 1000)}.wav"

                                    def save_wav():
                                        sf.write(temp_file, audio_float, sample_rate)

                                    await loop.run_in_executor(None, save_wav)

                                    # Send status update
                                    await websocket.send_json({
                                        "type": "status",
                                        "message": f"Processing {chunk_interval * 3}s chunk..."
                                    })

                                    # Transcribe accumulated chunk
                                    def transcribe_chunk():
                                        try:
                                            results = self.model.transcribe(
                                                audio=temp_file,
                                                language=language if language and language != 'auto' else None,
                                                return_time_stamps=True
                                            )
                                            return results, None
                                        except Exception as e:
                                            import traceback
                                            return None, str(e) + "\n" + traceback.format_exc()

                                    results, error = await loop.run_in_executor(None, transcribe_chunk)

                                    if error:
                                        print(f"Transcription error: {error}")
                                        await websocket.send_json({
                                            "type": "status",
                                            "message": "Processing audio..."
                                        })
                                    elif results and len(results) > 0:
                                        result = results[0]

                                        # Send timestamp results
                                        if hasattr(result, 'time_stamps') and result.time_stamps:
                                            for ts in result.time_stamps:
                                                if ts.text.strip():
                                                    # Add offset to timestamps
                                                    adjusted_start = total_duration + ts.start_time
                                                    adjusted_end = total_duration + ts.end_time

                                                    seg_data = {
                                                        "text": ts.text,
                                                        "start": adjusted_start,
                                                        "end": adjusted_end,
                                                        "speaker": "Unknown"
                                                    }
                                                    all_segments.append(seg_data)

                                                    await websocket.send_json({
                                                        "type": "transcription",
                                                        "text": ts.text,
                                                        "start": adjusted_start,
                                                        "end": adjusted_end,
                                                        "speaker": "Unknown"
                                                    })
                                        else:
                                            # Fallback: send text without precise timestamps
                                            if result.text.strip():
                                                seg_data = {
                                                    "text": result.text,
                                                    "start": total_duration,
                                                    "end": total_duration + (len(chunk_buffer) / sample_rate),
                                                    "speaker": "Unknown"
                                                }
                                                all_segments.append(seg_data)

                                                await websocket.send_json({
                                                    "type": "transcription",
                                                    "text": result.text,
                                                    "start": seg_data["start"],
                                                    "end": seg_data["end"],
                                                    "speaker": "Unknown"
                                                })

                                        print(f"Sent Qwen transcription for {chunk_interval} chunks")
                                    else:
                                        print("No speech detected in accumulated chunks")
                                        await websocket.send_json({
                                            "type": "status",
                                            "message": "No speech detected"
                                        })

                                    # Update total duration
                                    total_duration += len(chunk_buffer) / sample_rate

                                    # Reset buffer
                                    chunk_buffer = []
                                    chunk_count = 0

                                except Exception as e:
                                    import traceback
                                    error_msg = f"Qwen live transcription error: {e}\n{traceback.format_exc()}"
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
            print(f"Qwen live recording ended. Diarization enabled: {enable_diarization}")

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
                    full_audio_file = f"temp_qwen_complete_{int(time.time() * 1000)}.wav"

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
            print("Qwen live transcription session ended")
