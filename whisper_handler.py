import os
import asyncio
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import time
import queue
from threading import Event

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Warning: pyannote.audio not available. Speaker diarization will be disabled.")

class WhisperHandler:
    MODELS = {
        "tiny": {"ram": "4GB", "size": "tiny"},
        "base": {"ram": "5GB", "size": "base"},
        "small": {"ram": "6GB", "size": "small"},
        "medium": {"ram": "10GB", "size": "medium"},
        "large-v2": {"ram": "14GB", "size": "large-v2"},
        "large-v3": {"ram": "14+GB", "size": "large-v3"},
    }

    def __init__(self):
        self.model: Optional[WhisperModel] = None
        self.current_model_name: Optional[str] = None
        self.diarization_pipeline: Optional[Pipeline] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.model_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    def get_available_models(self) -> Dict:
        """Get list of available models with their status"""
        models = {}
        for name, info in self.MODELS.items():
            models[name] = {
                "ram": info["ram"],
                "installed": self._is_model_installed(name),
                "loaded": self.current_model_name == name
            }
        return models

    def _is_model_installed(self, model_name: str) -> bool:
        """Check if a model is already downloaded"""
        # faster-whisper stores models in the cache directory
        # This is a simplified check - actual implementation may vary
        model_path = self.model_cache_dir / f"models--Systran--faster-whisper-{model_name}"
        return model_path.exists()

    def refresh_installed_models(self):
        """Refresh the cached status of installed models"""
        # Force re-check of installed models
        pass

    async def _download_model_with_progress(self, model_name: str, websocket=None):
        """Download model with progress reporting"""
        try:
            if websocket:
                await websocket.send_json({
                    "type": "download_progress",
                    "status": "starting",
                    "message": f"Starting download of {model_name} model..."
                })

            # Load model (this will download if not present)
            # Note: faster-whisper doesn't provide granular download progress
            # For production, you might want to implement custom download logic
            start_time = time.time()

            def load_model():
                return WhisperModel(
                    model_name,
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
                    "message": f"Model loaded in {elapsed:.2f} seconds"
                })

            self.current_model_name = model_name
            return True

        except Exception as e:
            if websocket:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Failed to load model: {str(e)}"
                })
            raise

    async def load_model(self, model_name: str, websocket=None):
        """Load a Whisper model"""
        if self.current_model_name == model_name and self.model is not None:
            return  # Model already loaded

        # Unload current model if any
        self.unload_model()

        # Download/load new model
        await self._download_model_with_progress(model_name, websocket)

    def _load_diarization_pipeline(self):
        """Load speaker diarization pipeline"""
        if not PYANNOTE_AVAILABLE:
            return None

        if self.diarization_pipeline is None:
            try:
                # Note: This requires HuggingFace authentication token
                # Users need to set HF_TOKEN environment variable
                hf_token = os.getenv("HF_TOKEN")
                if hf_token:
                    # Login to HuggingFace hub first (modern approach)
                    try:
                        from huggingface_hub import login
                        login(token=hf_token, add_to_git_credential=False)
                        print("Successfully logged in to HuggingFace Hub")
                    except Exception as e:
                        print(f"Warning: Could not login to HuggingFace Hub: {e}")

                    # Load pipeline without passing token (it's already logged in)
                    try:
                        print("Loading speaker diarization pipeline (this may take a moment)...")
                        self.diarization_pipeline = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1"
                        )
                        if self.device == "cuda":
                            self.diarization_pipeline.to(torch.device("cuda"))
                        print("Successfully loaded speaker diarization pipeline")
                    except Exception as e:
                        error_msg = str(e)
                        print(f"Warning: Could not load diarization pipeline: {error_msg}")

                        # Check if it's a permissions/gated model issue
                        if "gated" in error_msg.lower() or "private" in error_msg.lower() or "authenticate" in error_msg.lower():
                            print("\nYou need to accept the terms for these HuggingFace models:")
                            print("1. https://huggingface.co/pyannote/speaker-diarization-3.1")
                            print("2. https://huggingface.co/pyannote/segmentation-3.0")
                            print("3. https://huggingface.co/pyannote/segmentation (if required)")
                            print("\nVisit each URL, click 'Agree and access repository', then restart the app.")

                        print("Speaker diarization will be disabled.")
                        self.diarization_pipeline = None
                else:
                    print("Warning: HF_TOKEN not set. Speaker diarization will be disabled.")
            except Exception as e:
                print(f"Warning: Could not load diarization pipeline: {e}")
                self.diarization_pipeline = None

        return self.diarization_pipeline

    def unload_model(self):
        """Unload Whisper model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
            self.current_model_name = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def unload_all(self):
        """Unload both model and diarization pipeline"""
        self.unload_model()
        if self.diarization_pipeline is not None:
            del self.diarization_pipeline
            self.diarization_pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def transcribe_file_streaming(self, file_path: str, model_name: str, language: str = None):
        """Transcribe an audio file with streaming updates"""
        # Load model if needed
        await self.load_model(model_name)

        yield {"type": "status", "message": "Model loaded, starting transcription..."}

        # Start transcription in background
        loop = asyncio.get_event_loop()

        def transcribe_worker():
            """Worker function to run transcription in thread"""
            return self.model.transcribe(
                file_path,
                beam_size=5,
                language=language if language and language != 'auto' else None,  # Auto-detect if None
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
        if PYANNOTE_AVAILABLE:
            pipeline = self._load_diarization_pipeline()
            if pipeline:
                try:
                    diarization = pipeline(file_path)

                    # Map speakers to segments
                    for i, segment in enumerate(transcription_segments):
                        start_time = segment["start"]
                        end_time = segment["end"]
                        mid_time = (start_time + end_time) / 2

                        # Find speaker at segment midpoint
                        speaker_found = False
                        for turn, _, speaker in diarization.itertracks(yield_label=True):
                            if turn.start <= mid_time <= turn.end:
                                transcription_segments[i]["speaker"] = f"Speaker {speaker}"
                                speaker_found = True
                                break

                        # If no speaker found at midpoint, try overlap
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
                                transcription_segments[i]["speaker"] = f"Speaker {best_speaker}"

                except Exception as e:
                    print(f"Diarization failed: {e}")
                    yield {"type": "warning", "message": f"Speaker diarization failed: {str(e)}"}

        # Merge consecutive segments from the same speaker
        merged_segments = []
        for segment in transcription_segments:
            if merged_segments and merged_segments[-1]["speaker"] == segment["speaker"]:
                # Same speaker - merge with previous segment
                merged_segments[-1]["end"] = segment["end"]
                merged_segments[-1]["text"] += " " + segment["text"]
            else:
                # New speaker - add new segment
                merged_segments.append(segment.copy())

        # Update full text from merged segments
        full_text = " ".join(seg["text"] for seg in merged_segments)

        # Send final result
        yield {
            "type": "complete",
            "language": info.language if hasattr(info, 'language') else "unknown",
            "segments": merged_segments,
            "full_text": full_text.strip()
        }

    async def transcribe_file(self, file_path: str, model_name: str, language: str = None) -> Dict:
        """Transcribe an audio file with speaker diarization"""
        # Load model if needed
        await self.load_model(model_name)

        # Transcribe
        segments, info = self.model.transcribe(
            file_path,
            beam_size=5,
            language=language if language and language != 'auto' else None,  # Auto-detect if None
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        # Collect segments
        transcription_segments = []
        full_text = ""

        for segment in segments:
            transcription_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "speaker": "Unknown"  # Will be updated with diarization
            })
            full_text += segment.text + " "

        # Apply speaker diarization if available
        if PYANNOTE_AVAILABLE:
            pipeline = self._load_diarization_pipeline()
            if pipeline:
                try:
                    print("Running speaker diarization...")
                    diarization = pipeline(file_path)

                    # Map speakers to segments
                    for i, segment in enumerate(transcription_segments):
                        start_time = segment["start"]
                        end_time = segment["end"]
                        mid_time = (start_time + end_time) / 2

                        # Find speaker at segment midpoint
                        speaker_found = False
                        for turn, _, speaker in diarization.itertracks(yield_label=True):
                            if turn.start <= mid_time <= turn.end:
                                transcription_segments[i]["speaker"] = f"Speaker {speaker}"
                                speaker_found = True
                                break

                        # If no speaker found at midpoint, try overlap
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
                                transcription_segments[i]["speaker"] = f"Speaker {best_speaker}"

                    print("Speaker diarization completed")
                except Exception as e:
                    print(f"Diarization failed: {e}")

        # Merge consecutive segments from the same speaker
        merged_segments = []
        for segment in transcription_segments:
            if merged_segments and merged_segments[-1]["speaker"] == segment["speaker"]:
                # Same speaker - merge with previous segment
                merged_segments[-1]["end"] = segment["end"]
                merged_segments[-1]["text"] += " " + segment["text"]
            else:
                # New speaker - add new segment
                merged_segments.append(segment.copy())

        # Update full text from merged segments
        full_text = " ".join(seg["text"] for seg in merged_segments)

        return {
            "language": info.language if hasattr(info, 'language') else "unknown",
            "segments": merged_segments,
            "full_text": full_text.strip()
        }

    async def transcribe_live(self, websocket):
        """Perform live transcription from browser audio stream"""
        audio_chunks = []
        chunk_count = 0
        loop = asyncio.get_event_loop()
        model_loaded = False
        language = None
        enable_diarization = True

        # Storage for full recording
        all_audio_data = []
        all_segments = []
        total_duration = 0.0  # Track total audio duration for timestamp offset

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
                        print(f"Live recording initialized with model: {model_name}, language: {language}, diarization: {enable_diarization}")

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
                                        # Convert generator to list
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
                                        if segment.text.strip():  # Only send non-empty text
                                            # Add offset to timestamps based on total duration so far
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

                                    # Update total duration after processing chunk
                                    total_duration += chunk_duration
                                else:
                                    print("No speech detected in chunk")
                                    await websocket.send_json({
                                        "type": "status",
                                        "message": "No speech detected"
                                    })
                                    # Still update duration even if no speech
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
            print(f"Live recording ended. Diarization enabled: {enable_diarization}, Audio data: {len(all_audio_data)} samples, Segments: {len(all_segments)}")

            # Helper function to safely send messages
            async def safe_send(message):
                try:
                    await websocket.send_json(message)
                    return True
                except Exception as e:
                    print(f"Could not send message (WebSocket closed): {e}")
                    return False

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

                        # Map speakers to segments
                        for segment in all_segments:
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

                            # If no speaker found at midpoint, try overlap
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

                        # Merge consecutive same-speaker segments
                        merged_segments = []
                        for segment in all_segments:
                            if merged_segments and merged_segments[-1]["speaker"] == segment["speaker"]:
                                merged_segments[-1]["end"] = segment["end"]
                                merged_segments[-1]["text"] += " " + segment["text"]
                            else:
                                merged_segments.append(segment.copy())

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
            else:
                if not enable_diarization:
                    print("Speaker diarization disabled by user")
                elif not all_audio_data:
                    print("No audio data accumulated for diarization")
                elif not all_segments:
                    print("No segments to diarize")

            # Send final status
            await safe_send({
                "type": "status",
                "message": "Recording stopped"
            })
            print("Live transcription session ended")
