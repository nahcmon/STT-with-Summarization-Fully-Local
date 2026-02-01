import os
import asyncio
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import soundfile as sf
import time

try:
    from pyannote.audio import Model, Inference
    from pyannote.audio.pipelines import SpeakerDiarization
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Warning: pyannote.audio not available. TSE will be disabled.")


class TSEHandler:
    """Target Speaker Extraction handler using pyannote.audio embeddings"""

    def __init__(self):
        self.enrollment_embedding: Optional[np.ndarray] = None
        self.speaker_encoder: Optional[Inference] = None
        self.diarization_pipeline: Optional[SpeakerDiarization] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enrollment_metadata = None  # Store enrollment audio metadata

    async def load_speaker_encoder(self):
        """Load pyannote speaker embedding model"""
        if not PYANNOTE_AVAILABLE:
            raise RuntimeError("pyannote.audio is not available")

        if self.speaker_encoder is None:
            try:
                # Check for HF_TOKEN
                hf_token = os.getenv("HF_TOKEN")
                if not hf_token:
                    raise RuntimeError("HF_TOKEN environment variable not set. Speaker extraction requires HuggingFace authentication.")

                # Login to HuggingFace hub
                try:
                    from huggingface_hub import login
                    login(token=hf_token, add_to_git_credential=False)
                    print("Successfully logged in to HuggingFace Hub")
                except Exception as e:
                    print(f"Warning: Could not login to HuggingFace Hub: {e}")

                print("Loading speaker embedding model (this may take a moment)...")

                # Load speaker embedding model (wespeaker-voxceleb-resnet34-LM)
                # This is the same model used internally by pyannote diarization
                model = Model.from_pretrained(
                    "pyannote/wespeaker-voxceleb-resnet34-LM"
                )

                # Create inference wrapper
                self.speaker_encoder = Inference(
                    model,
                    window="whole"  # Process entire audio file at once
                )

                if self.device == "cuda":
                    self.speaker_encoder.to(torch.device("cuda"))

                print("Successfully loaded speaker embedding model")

            except Exception as e:
                error_msg = str(e)
                print(f"Error loading speaker encoder: {error_msg}")

                if "gated" in error_msg.lower() or "private" in error_msg.lower():
                    print("\nYou need to accept the terms for these HuggingFace models:")
                    print("1. https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM")
                    print("2. https://huggingface.co/pyannote/speaker-diarization-3.1")
                    print("3. https://huggingface.co/pyannote/segmentation-3.0")
                    print("\nVisit each URL, click 'Agree and access repository', then restart the app.")

                raise

    def _load_diarization_pipeline(self):
        """Load speaker diarization pipeline (shared with WhisperHandler logic)"""
        if not PYANNOTE_AVAILABLE:
            return None

        if self.diarization_pipeline is None:
            try:
                hf_token = os.getenv("HF_TOKEN")
                if hf_token:
                    try:
                        from huggingface_hub import login
                        login(token=hf_token, add_to_git_credential=False)
                    except Exception as e:
                        print(f"Warning: Could not login to HuggingFace Hub: {e}")

                    try:
                        print("Loading speaker diarization pipeline...")
                        from pyannote.audio import Pipeline
                        self.diarization_pipeline = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1"
                        )
                        if self.device == "cuda":
                            self.diarization_pipeline.to(torch.device("cuda"))
                        print("Successfully loaded speaker diarization pipeline")
                    except Exception as e:
                        print(f"Warning: Could not load diarization pipeline: {e}")
                        self.diarization_pipeline = None
                else:
                    print("Warning: HF_TOKEN not set. Speaker diarization will be disabled.")
            except Exception as e:
                print(f"Warning: Could not load diarization pipeline: {e}")
                self.diarization_pipeline = None

        return self.diarization_pipeline

    async def process_enrollment(self, audio_path: str) -> dict:
        """Extract speaker embedding from enrollment sample"""
        await self.load_speaker_encoder()

        loop = asyncio.get_event_loop()

        def extract_embedding():
            """Extract embedding in thread"""
            # Load audio file metadata
            audio_info = sf.info(audio_path)
            duration = audio_info.duration

            # Extract embedding
            embedding = self.speaker_encoder(audio_path)

            return embedding, duration

        try:
            # Run embedding extraction in executor
            embedding, duration = await loop.run_in_executor(None, extract_embedding)

            # Store enrollment embedding
            self.enrollment_embedding = embedding
            self.enrollment_metadata = {
                "duration": duration,
                "path": audio_path
            }

            print(f"Enrollment embedding extracted: shape {embedding.shape}, duration {duration:.2f}s")

            return {
                "status": "success",
                "duration": duration,
                "embedding_shape": str(embedding.shape)
            }

        except Exception as e:
            print(f"Enrollment processing error: {e}")
            raise

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        # Ensure embeddings are 1D
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()

        # Calculate cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def extract_speaker_streaming(
        self,
        audio_path: str,
        whisper_handler,
        whisper_model: str,
        language: str = None,
        similarity_threshold: float = 0.75
    ):
        """
        Extract target speaker from mixed audio using embedding similarity

        Args:
            audio_path: Path to mixed audio file
            whisper_handler: WhisperHandler instance for transcription
            whisper_model: Whisper model name
            language: Language for transcription
            similarity_threshold: Minimum cosine similarity to consider a match (0-1)
        """
        if self.enrollment_embedding is None:
            raise RuntimeError("No enrollment sample loaded. Please upload enrollment audio first.")

        # Ensure speaker encoder is loaded
        await self.load_speaker_encoder()

        loop = asyncio.get_event_loop()

        # Step 1: Load Whisper model
        yield {"type": "status", "message": "Loading Whisper model..."}
        await whisper_handler.load_model(whisper_model)

        # Step 2: Transcribe full audio
        yield {"type": "status", "message": "Transcribing audio..."}

        def transcribe_worker():
            """Worker function to run transcription in thread"""
            return whisper_handler.model.transcribe(
                audio_path,
                beam_size=5,
                language=language if language and language != 'auto' else None,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

        segments_generator, info = await loop.run_in_executor(None, transcribe_worker)

        # Collect all segments
        all_segments = []
        def collect_segments(gen):
            return list(gen)

        segment_list = await loop.run_in_executor(None, collect_segments, segments_generator)

        for segment in segment_list:
            all_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "speaker": "Unknown"
            })

        yield {
            "type": "progress",
            "message": f"Found {len(all_segments)} speech segments",
            "segments_count": len(all_segments)
        }

        # Step 3: Run diarization to get speaker labels
        yield {"type": "status", "message": "Running speaker diarization..."}

        pipeline = self._load_diarization_pipeline()
        if not pipeline:
            raise RuntimeError("Speaker diarization pipeline not available. Cannot extract speakers.")

        def run_diarization():
            return pipeline(audio_path)

        diarization = await loop.run_in_executor(None, run_diarization)

        # Map speakers to segments
        for i, segment in enumerate(all_segments):
            start_time = segment["start"]
            end_time = segment["end"]
            mid_time = (start_time + end_time) / 2

            # Find speaker at segment midpoint
            speaker_found = False
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start <= mid_time <= turn.end:
                    all_segments[i]["speaker"] = f"SPEAKER_{speaker}"
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
                    all_segments[i]["speaker"] = f"SPEAKER_{best_speaker}"

        # Step 4: Extract embeddings for each unique speaker
        yield {"type": "status", "message": "Extracting speaker embeddings..."}

        # Get unique speakers
        speakers = set(seg["speaker"] for seg in all_segments if seg["speaker"] != "Unknown")

        # Extract audio segments for each speaker and get embeddings
        speaker_embeddings = {}
        speaker_similarities = {}

        # Load audio file
        audio_data, sample_rate = sf.read(audio_path)

        for speaker in speakers:
            # Find all segments for this speaker
            speaker_segments = [seg for seg in all_segments if seg["speaker"] == speaker]

            # Extract audio for this speaker (concatenate all segments)
            speaker_audio = []
            for seg in speaker_segments:
                start_sample = int(seg["start"] * sample_rate)
                end_sample = int(seg["end"] * sample_rate)
                speaker_audio.append(audio_data[start_sample:end_sample])

            if speaker_audio:
                # Concatenate all segments
                speaker_audio_combined = np.concatenate(speaker_audio)

                # Save to temp file for embedding extraction
                temp_speaker_file = f"temp_speaker_{speaker}_{int(time.time() * 1000)}.wav"

                try:
                    def save_speaker_wav():
                        sf.write(temp_speaker_file, speaker_audio_combined, sample_rate)

                    await loop.run_in_executor(None, save_speaker_wav)

                    # Extract embedding
                    def extract_speaker_embedding():
                        return self.speaker_encoder(temp_speaker_file)

                    embedding = await loop.run_in_executor(None, extract_speaker_embedding)
                    speaker_embeddings[speaker] = embedding

                    # Calculate similarity with enrollment
                    similarity = self._cosine_similarity(self.enrollment_embedding, embedding)
                    speaker_similarities[speaker] = similarity

                    print(f"Speaker {speaker}: similarity = {similarity:.3f}")

                finally:
                    # Clean up temp file
                    if os.path.exists(temp_speaker_file):
                        try:
                            os.remove(temp_speaker_file)
                        except Exception:
                            pass

        # Step 5: Find best matching speaker
        if not speaker_similarities:
            raise RuntimeError("No speakers found in audio")

        best_speaker = max(speaker_similarities.items(), key=lambda x: x[1])
        best_speaker_id, best_similarity = best_speaker

        yield {
            "type": "progress",
            "message": f"Best match: {best_speaker_id} (similarity: {best_similarity:.2f})",
            "best_speaker": best_speaker_id,
            "similarity": best_similarity,
            "all_speakers": {k: float(v) for k, v in speaker_similarities.items()}
        }

        # Check if best match meets threshold
        if best_similarity < similarity_threshold:
            yield {
                "type": "warning",
                "message": f"Low confidence: Best match similarity ({best_similarity:.2f}) is below threshold ({similarity_threshold})"
            }

        # Step 6: Filter segments for target speaker
        target_segments = [seg for seg in all_segments if seg["speaker"] == best_speaker_id]

        yield {
            "type": "status",
            "message": f"Extracting {len(target_segments)} segments from target speaker..."
        }

        # Step 7: Extract audio for target speaker
        target_audio_segments = []
        for seg in target_segments:
            start_sample = int(seg["start"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)
            target_audio_segments.append(audio_data[start_sample:end_sample])

        if not target_audio_segments:
            raise RuntimeError("No audio segments found for target speaker")

        # Concatenate all target segments
        target_audio = np.concatenate(target_audio_segments)

        # Save extracted audio to temp file
        extracted_audio_file = f"temp_extracted_{int(time.time() * 1000)}.wav"

        def save_extracted_audio():
            sf.write(extracted_audio_file, target_audio, sample_rate)

        await loop.run_in_executor(None, save_extracted_audio)

        # Calculate total duration
        total_duration = len(target_audio) / sample_rate

        # Merge consecutive segments from target speaker
        merged_segments = []
        for segment in target_segments:
            segment_copy = segment.copy()
            segment_copy["speaker"] = "Target Speaker"

            if merged_segments and (segment["start"] - merged_segments[-1]["end"]) < 0.5:
                # Merge if gap is less than 0.5 seconds
                merged_segments[-1]["end"] = segment["end"]
                merged_segments[-1]["text"] += " " + segment["text"]
            else:
                merged_segments.append(segment_copy)

        # Generate full text
        full_text = " ".join(seg["text"] for seg in merged_segments)

        # Step 8: Return final result
        yield {
            "type": "complete",
            "language": info.language if hasattr(info, 'language') else "unknown",
            "segments": merged_segments,
            "full_text": full_text.strip(),
            "extracted_audio_path": extracted_audio_file,
            "duration": total_duration,
            "similarity": float(best_similarity),
            "speaker_id": best_speaker_id,
            "total_segments": len(merged_segments),
            "all_speakers": {k: float(v) for k, v in speaker_similarities.items()}
        }

    def unload_all(self):
        """Cleanup resources"""
        self.enrollment_embedding = None
        self.enrollment_metadata = None

        if self.speaker_encoder is not None:
            del self.speaker_encoder
            self.speaker_encoder = None

        if self.diarization_pipeline is not None:
            del self.diarization_pipeline
            self.diarization_pipeline = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
