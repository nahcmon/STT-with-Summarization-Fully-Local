import os
import warnings
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi import Request
import uvicorn
from pathlib import Path
import shutil
from datetime import datetime
import time
import asyncio
import json

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_fabric")
warnings.filterwarnings("ignore", category=FutureWarning, module="lightning_fabric")
warnings.filterwarnings("ignore", message=".*weights_only.*")

# Suppress MP3 decoding warnings from FFmpeg/mpg123
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
# Redirect stderr to suppress FFmpeg warnings
import sys
if sys.platform == 'win32':
    # On Windows, suppress FFmpeg console output
    os.environ['FFREPORT'] = 'level=quiet'

from transcription_handler import TranscriptionHandler
from llm_handler import LLMHandler
from tse_handler import TSEHandler

# Initialize handlers
transcription_handler = TranscriptionHandler()
llm_handler = LLMHandler()
tse_handler = TSEHandler()

# Backward compatibility: whisper_handler is now an alias
whisper_handler = transcription_handler.whisper

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    await llm_handler.close()
    tse_handler.unload_all()

app = FastAPI(lifespan=lifespan)

# Create necessary directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/transcription/models")
async def get_transcription_models():
    """Get available transcription models from all backends (Whisper, Qwen)"""
    models = transcription_handler.get_available_models()
    return JSONResponse(content=models)

# Backward compatibility: legacy /api/whisper/models endpoint
@app.get("/api/whisper/models")
async def get_whisper_models():
    """Get available Whisper models and their download status (legacy endpoint)"""
    all_models = transcription_handler.get_available_models()
    return JSONResponse(content=all_models.get("whisper", {}))

@app.post("/api/whisper/refresh")
async def refresh_whisper_models():
    """Refresh the list of installed Whisper models"""
    # Refresh is automatic on model check, so just return current status
    all_models = transcription_handler.get_available_models()
    return JSONResponse(content=all_models.get("whisper", {}))

@app.post("/api/transcription/unload")
async def unload_transcription_model():
    """Unload current transcription model from memory"""
    transcription_handler.unload_model()
    return JSONResponse(content={"status": "success", "message": "Model unloaded"})

@app.post("/api/whisper/unload")
async def unload_whisper_model():
    """Unload Whisper model from memory (legacy endpoint)"""
    transcription_handler.unload_model()
    return JSONResponse(content={"status": "success", "message": "Model unloaded"})

@app.post("/api/transcription/unload-all")
async def unload_all_models():
    """Unload all transcription models and pipelines from memory"""
    transcription_handler.unload_all()
    return JSONResponse(content={"status": "success", "message": "All models and pipelines unloaded"})

@app.post("/api/whisper/unload-all")
async def unload_all():
    """Unload Whisper model and pipeline from memory (legacy endpoint)"""
    transcription_handler.unload_all()
    return JSONResponse(content={"status": "success", "message": "Model and pipeline unloaded"})

@app.post("/api/transcribe/file")
async def transcribe_file(file: UploadFile = File(...), model: str = "whisper-base", language: str = "auto"):
    """Transcribe an uploaded audio file (supports both Whisper and Qwen models)"""
    import tempfile

    # Use temporary file that's automatically deleted
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
        temp_path = temp_file.name
        try:
            # Save uploaded file to temp location
            shutil.copyfileobj(file.file, temp_file)
            temp_file.flush()

            # Collect results from streaming transcription
            result = None
            async for update in transcription_handler.transcribe_file_streaming(temp_path, model, language):
                if update.get("type") == "complete":
                    result = update
                    break

            if result:
                return JSONResponse(content=result)
            else:
                raise HTTPException(status_code=500, detail="Transcription failed")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Always delete temp file
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.post("/api/transcribe/file/stream")
async def transcribe_file_stream(file: UploadFile = File(...), model: str = "whisper-base", language: str = "auto"):
    """Transcribe an uploaded audio file with streaming results (supports both Whisper and Qwen models)"""
    import tempfile

    async def generate():
        # Use temporary file that's automatically deleted
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            temp_path = temp_file.name
            try:
                # Save uploaded file to temp location
                shutil.copyfileobj(file.file, temp_file)
                temp_file.flush()

                # Send progress update
                yield f"data: {json.dumps({'type': 'status', 'message': 'Audio file uploaded, starting transcription...'})}\n\n"

                # Transcribe with streaming updates using unified handler
                async for update in transcription_handler.transcribe_file_streaming(temp_path, model, language):
                    yield f"data: {json.dumps(update)}\n\n"

                yield "data: {\"type\": \"done\"}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            finally:
                # Always delete temp file
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.websocket("/ws/transcribe/live")
async def transcribe_live(websocket: WebSocket):
    """WebSocket endpoint for live transcription (supports both Whisper and Qwen models)"""
    await websocket.accept()
    try:
        # Receive init message to get model selection
        init_msg = await websocket.receive_json()

        if init_msg.get("type") == "init":
            model_name = init_msg.get('model', 'whisper-base')
            chunk_interval = init_msg.get('chunkInterval', 1)  # Default: process immediately

            # Start live transcription using unified handler
            await transcription_handler.transcribe_live(websocket, model_name, chunk_interval)
        else:
            await websocket.send_json({"type": "error", "message": "Expected init message"})
            await websocket.close()

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass  # Connection might already be closed
        try:
            await websocket.close()
        except:
            pass  # Already closed

# TSE (Target Speaker Extraction) Routes
@app.post("/api/tse/enroll")
async def enroll_speaker(file: UploadFile = File(...)):
    """Upload and process enrollment sample for target speaker extraction"""
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
        temp_path = temp_file.name
        try:
            # Save uploaded file to temp location
            shutil.copyfileobj(file.file, temp_file)
            temp_file.flush()

            # Process enrollment
            result = await tse_handler.process_enrollment(temp_path)

            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Always delete temp file
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.post("/api/tse/extract/stream")
async def extract_speaker_stream(
    file: UploadFile = File(...),
    model: str = "base",
    language: str = "auto",
    threshold: float = 0.75
):
    """Extract target speaker from mixed audio with streaming results"""
    import tempfile

    async def generate():
        # Use temporary file that's automatically deleted
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            temp_path = temp_file.name
            extracted_audio_path = None
            try:
                # Save uploaded file to temp location
                shutil.copyfileobj(file.file, temp_file)
                temp_file.flush()

                # Send progress update
                yield f"data: {json.dumps({'type': 'status', 'message': 'Audio file uploaded, starting speaker extraction...'})}\n\n"

                # Extract speaker with streaming updates
                async for update in tse_handler.extract_speaker_streaming(
                    temp_path,
                    whisper_handler,
                    model,
                    language,
                    threshold
                ):
                    # Store extracted audio path for cleanup
                    if update.get("type") == "complete":
                        extracted_audio_path = update.get("extracted_audio_path")
                        # Serve the extracted audio file
                        if extracted_audio_path and os.path.exists(extracted_audio_path):
                            # Copy to static/temp directory for serving
                            temp_static_dir = Path("static/temp")
                            temp_static_dir.mkdir(exist_ok=True)
                            filename = f"extracted_{int(time.time() * 1000)}.wav"
                            static_path = temp_static_dir / filename
                            shutil.copy(extracted_audio_path, static_path)
                            update["audio_url"] = f"/static/temp/{filename}"
                            # Remove the path from response
                            del update["extracted_audio_path"]

                    yield f"data: {json.dumps(update)}\n\n"

                yield "data: {\"type\": \"done\"}\n\n"

            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"TSE error: {error_details}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            finally:
                # Always delete temp files
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
                if extracted_audio_path and os.path.exists(extracted_audio_path):
                    try:
                        os.remove(extracted_audio_path)
                    except Exception:
                        pass

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/api/lm-studio/connect")
async def connect_lm_studio(request: Request):
    """Connect to LM Studio server"""
    data = await request.json()
    server_url = data.get("server_url")

    try:
        await llm_handler.connect(server_url)
        return JSONResponse(content={"status": "success", "message": "Connected to LM Studio"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/lm-studio/models")
async def get_lm_studio_models():
    """Get available models from LM Studio"""
    try:
        models = await llm_handler.get_models()
        return JSONResponse(content={"models": models})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/lm-studio/process")
async def process_with_llm(request: Request):
    """Send transcribed text to LLM"""
    data = await request.json()
    text = data.get("text", "")
    model = data.get("model", "")
    system_prompt = data.get("system_prompt", "You are a helpful assistant that summarizes transcribed conversations.")

    try:
        result = await llm_handler.process_text(text, model, system_prompt)
        return JSONResponse(content={"result": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/lm-studio/process/stream")
async def process_with_llm_stream(request: Request):
    """Send transcribed text to LLM with streaming response"""
    data = await request.json()
    text = data.get("text", "")
    model = data.get("model", "")
    system_prompt = data.get("system_prompt", "You are a helpful assistant that summarizes transcribed conversations.")

    async def generate():
        try:
            async for chunk in llm_handler.process_text_streaming(text, model, system_prompt):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3456)
