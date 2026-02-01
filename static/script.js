// Global state
let currentTranscription = '';
let websocket = null;
let isRecording = false;

// DOM Elements
const whisperModelSelect = document.getElementById('whisper-model');
const languageSelect = document.getElementById('language-select');
const refreshWhisperBtn = document.getElementById('refresh-whisper');
const unloadModelBtn = document.getElementById('unload-model');
const unloadAllBtn = document.getElementById('unload-all');
const whisperStatus = document.getElementById('whisper-status');
const liveDiarizationCheckbox = document.getElementById('live-diarization');

const audioFileInput = document.getElementById('audio-file');
const transcribeFileBtn = document.getElementById('transcribe-file');

const startRecordingBtn = document.getElementById('start-recording');
const stopRecordingBtn = document.getElementById('stop-recording');
const recordingStatus = document.getElementById('recording-status');

const progressSection = document.getElementById('progress-section');
const progressFill = document.getElementById('progress-fill');
const progressDetails = document.getElementById('progress-details');

const transcriptionOutput = document.getElementById('transcription-output');
const segmentsOutput = document.getElementById('segments-output');

const lmStudioUrl = document.getElementById('lm-studio-url');
const connectLMStudioBtn = document.getElementById('connect-lm-studio');
const lmStudioStatus = document.getElementById('lm-studio-status');
const llmModelSelect = document.getElementById('llm-model');
const refreshLLMBtn = document.getElementById('refresh-llm');
const systemPrompt = document.getElementById('system-prompt');
const processLLMBtn = document.getElementById('process-llm');
const llmOutput = document.getElementById('llm-output');
const llmResult = document.getElementById('llm-result');

// TSE elements
const enrollmentAudioInput = document.getElementById('enrollment-audio');
const uploadEnrollmentBtn = document.getElementById('upload-enrollment');
const enrollmentStatus = document.getElementById('enrollment-status');
const mixedAudioInput = document.getElementById('mixed-audio');
const extractSpeakerBtn = document.getElementById('extract-speaker');
const extractionStatus = document.getElementById('extraction-status');
const similarityThreshold = document.getElementById('similarity-threshold');
const thresholdValue = document.getElementById('threshold-value');
const tseResults = document.getElementById('tse-results');

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;

        // Update buttons
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Update content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tab}-tab`).classList.add('active');
    });
});

// Show/hide Qwen chunk settings based on model selection
whisperModelSelect.addEventListener('change', function() {
    const isQwen = this.value.startsWith('qwen-');
    const qwenSettings = document.getElementById('qwen-chunk-settings');
    if (qwenSettings) {
        qwenSettings.style.display = isQwen ? 'block' : 'none';
    }
});

// Helper functions
function showStatus(element, message, type = 'info') {
    element.textContent = message;
    element.className = `status-${type}`;
    element.style.display = 'block';
}

function showProgress(show, details = '') {
    progressSection.style.display = show ? 'block' : 'none';
    if (details) {
        progressDetails.textContent = details;
    }
}

function updateProgress(percent) {
    progressFill.style.width = `${percent}%`;
    progressFill.textContent = `${percent}%`;
}

// Transcription model management
async function refreshWhisperModels() {
    try {
        const response = await fetch('/api/transcription/models');
        const data = await response.json();

        // Combine Whisper and Qwen models
        const allModels = {...data.whisper, ...data.qwen};

        // Update select options with status indicators
        Array.from(whisperModelSelect.options).forEach(option => {
            const modelName = option.value;
            if (allModels[modelName]) {
                const model = allModels[modelName];
                let statusText = '';
                if (model.loaded) {
                    statusText = ' [LOADED]';
                } else if (model.installed) {
                    statusText = ' [INSTALLED]';
                }
                // Remove old status and add new one
                const baseText = option.textContent.split(' [')[0];
                option.textContent = `${baseText}${statusText}`;
            }
        });

        showStatus(whisperStatus, 'Models refreshed', 'success');
        setTimeout(() => whisperStatus.style.display = 'none', 3000);
    } catch (error) {
        showStatus(whisperStatus, `Error: ${error.message}`, 'error');
    }
}

async function unloadWhisperModel() {
    try {
        const response = await fetch('/api/whisper/unload', { method: 'POST' });
        const data = await response.json();
        showStatus(whisperStatus, data.message, 'success');
        await refreshWhisperModels();
    } catch (error) {
        showStatus(whisperStatus, `Error: ${error.message}`, 'error');
    }
}

async function unloadAll() {
    try {
        const response = await fetch('/api/whisper/unload-all', { method: 'POST' });
        const data = await response.json();
        showStatus(whisperStatus, data.message, 'success');
        await refreshWhisperModels();
    } catch (error) {
        showStatus(whisperStatus, `Error: ${error.message}`, 'error');
    }
}

// File transcription with streaming
async function transcribeFile() {
    const file = audioFileInput.files[0];
    if (!file) {
        alert('Please select an audio file');
        return;
    }

    const model = whisperModelSelect.value;
    const formData = new FormData();
    formData.append('file', file);

    try {
        transcribeFileBtn.disabled = true;
        clearTranscription();
        showProgress(true, 'Uploading audio file...');
        updateProgress(10);

        // Add streaming indicator
        transcriptionOutput.innerHTML = '<div class="streaming"><span class="streaming-indicator"></span> Preparing transcription...</div>';

        const language = languageSelect.value;
        const response = await fetch(`/api/transcribe/file/stream?model=${model}&language=${language}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        // Handle streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let partialSegments = [];

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.trim() && line.startsWith('data: ')) {
                    try {
                        const dataStr = line.slice(6);
                        const data = JSON.parse(dataStr);

                        console.log('Streaming update:', data.type); // Debug

                        if (data.type === 'status') {
                            showProgress(true, data.message);
                            if (data.message.includes('diarization')) {
                                updateProgress(70);
                            } else if (data.message.includes('loaded')) {
                                updateProgress(20);
                            } else {
                                updateProgress(30);
                            }
                        } else if (data.type === 'partial_segment') {
                            partialSegments.push(data.segment);
                            // Show partial results in real-time
                            displayPartialTranscription(partialSegments);
                            const progress = Math.min(30 + (data.progress * 0.4), 69);
                            updateProgress(progress);
                        } else if (data.type === 'complete') {
                            displayTranscription(data);
                            updateProgress(100);
                        } else if (data.type === 'warning') {
                            console.warn(data.message);
                        } else if (data.type === 'error') {
                            throw new Error(data.message);
                        } else if (data.type === 'done') {
                            setTimeout(() => showProgress(false), 1000);
                        }
                    } catch (e) {
                        console.error('Error parsing streaming data:', e);
                        // Continue processing other lines
                    }
                }
            }
        }
    } catch (error) {
        showProgress(false);
        alert(`Error: ${error.message}`);
    } finally {
        transcribeFileBtn.disabled = false;
    }
}

function displayPartialTranscription(segments) {
    // Show partial results as they come in with streaming indicator
    const fullText = segments.map(s => s.text).join(' ');
    transcriptionOutput.innerHTML = `
        <div class="streaming-container">
            <span class="streaming-indicator"></span>
            <p>${fullText}</p>
        </div>
    `;

    // Show segments with streaming animation
    segmentsOutput.innerHTML = '<h3><span class="streaming-indicator"></span> Transcribing...</h3>';
    segments.forEach(segment => {
        const segmentDiv = document.createElement('div');
        segmentDiv.className = 'segment streaming';
        const timeStr = `${formatTime(segment.start)} - ${formatTime(segment.end)}`;

        segmentDiv.innerHTML = `
            <div class="segment-header">
                <div class="segment-info">
                    <span class="segment-speaker">${segment.speaker}</span>
                    <span class="segment-time">${timeStr}</span>
                </div>
            </div>
            <div class="segment-text">${segment.text}</div>
        `;
        segmentsOutput.appendChild(segmentDiv);
    });

    // Auto scroll
    segmentsOutput.scrollTop = segmentsOutput.scrollHeight;
}

// Live transcription
let mediaRecorder = null;
let audioChunks = [];
let recordingStartTime = 0;
let audioContext = null;
let mediaStreamSource = null;
let processor = null;

async function startLiveRecording() {
    const model = whisperModelSelect.value;

    try {
        // Request microphone permission
        showStatus(recordingStatus, 'Requesting microphone permission...', 'info');
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            }
        });

        showStatus(recordingStatus, 'Microphone access granted', 'success');

        // Connect to WebSocket
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/transcribe/live`;

        websocket = new WebSocket(wsUrl);

        websocket.onopen = async () => {
            // Send model selection, language, diarization setting, and chunk interval
            const language = languageSelect.value;
            const enableDiarization = liveDiarizationCheckbox.checked;
            const chunkIntervalSelect = document.getElementById('chunk-interval');
            const chunkInterval = chunkIntervalSelect ? parseInt(chunkIntervalSelect.value) : 1;

            websocket.send(JSON.stringify({
                type: 'init',
                model,
                language,
                enableDiarization,
                chunkInterval
            }));

            // Setup Web Audio API for PCM capture
            audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });

            mediaStreamSource = audioContext.createMediaStreamSource(stream);

            // Create script processor to capture raw audio
            const bufferSize = 4096;
            processor = audioContext.createScriptProcessor(bufferSize, 1, 1);

            let audioBuffer = [];
            let lastSendTime = Date.now();
            const sendInterval = 3000; // Send every 3 seconds

            processor.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                audioBuffer.push(new Float32Array(inputData));

                // Send accumulated audio every 3 seconds
                const now = Date.now();
                if (now - lastSendTime >= sendInterval && audioBuffer.length > 0) {
                    // Combine all buffered audio
                    const totalLength = audioBuffer.reduce((acc, arr) => acc + arr.length, 0);
                    const combined = new Float32Array(totalLength);
                    let offset = 0;
                    for (const chunk of audioBuffer) {
                        combined.set(chunk, offset);
                        offset += chunk.length;
                    }

                    // Convert float32 to int16 PCM
                    const int16Array = new Int16Array(combined.length);
                    for (let i = 0; i < combined.length; i++) {
                        const s = Math.max(-1, Math.min(1, combined[i]));
                        int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                    }

                    console.log('Sending audio chunk:', int16Array.length, 'samples');

                    // Send PCM data
                    if (websocket && websocket.readyState === WebSocket.OPEN) {
                        websocket.send(JSON.stringify({
                            type: 'audio',
                            data: Array.from(int16Array),
                            sampleRate: 16000,
                            channels: 1,
                            timestamp: now - recordingStartTime
                        }));
                    }

                    // Clear buffer
                    audioBuffer = [];
                    lastSendTime = now;
                }
            };

            mediaStreamSource.connect(processor);
            processor.connect(audioContext.destination);

            recordingStartTime = Date.now();
            isRecording = true;
            startRecordingBtn.disabled = true;
            stopRecordingBtn.disabled = false;

            showStatus(recordingStatus, 'Recording... Speak now!', 'success');
            clearTranscription();

            // Store stream for cleanup
            window._audioStream = stream;
        };

        websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('Live recording message:', data.type, data); // Debug

            if (data.type === 'download_progress') {
                showProgress(true, data.message);
                if (data.status === 'complete') {
                    setTimeout(() => showProgress(false), 2000);
                }
            } else if (data.type === 'status') {
                console.log('Status update:', data.message); // Debug
                showStatus(recordingStatus, data.message, 'info');

                // Close WebSocket when server confirms recording stopped
                if (data.message === 'Recording stopped' && websocket) {
                    setTimeout(() => {
                        if (websocket) {
                            websocket.close();
                            websocket = null;
                        }
                    }, 500);
                    showStatus(recordingStatus, 'Recording stopped', 'success');
                    setTimeout(() => recordingStatus.style.display = 'none', 3000);
                }
            } else if (data.type === 'transcription') {
                console.log('Transcription received:', data.text); // Debug
                const speaker = data.speaker || 'Unknown';
                addLiveTranscription(data.text, data.start, data.end, speaker);
            } else if (data.type === 'diarization_complete') {
                console.log('Diarization complete, updating segments'); // Debug
                updateLiveSegmentsWithSpeakers(data.segments);
            } else if (data.type === 'warning') {
                console.warn('Warning:', data.message); // Debug
                showStatus(recordingStatus, data.message, 'warning');
            } else if (data.type === 'error') {
                console.error('Live recording error:', data.message); // Debug
                showStatus(recordingStatus, `Error: ${data.message}`, 'error');
                stopLiveRecording();
            }
        };

        websocket.onerror = (error) => {
            showStatus(recordingStatus, 'WebSocket error occurred', 'error');
            console.error('WebSocket error:', error);
        };

        websocket.onclose = () => {
            if (isRecording) {
                stopLiveRecording();
            }
        };

    } catch (error) {
        if (error.name === 'NotAllowedError') {
            showStatus(recordingStatus, 'Microphone permission denied', 'error');
        } else if (error.name === 'NotFoundError') {
            showStatus(recordingStatus, 'No microphone found', 'error');
        } else {
            showStatus(recordingStatus, `Error: ${error.message}`, 'error');
        }
        console.error('Microphone error:', error);
    }
}

function stopLiveRecording() {
    // Stop Web Audio API components
    if (processor) {
        processor.disconnect();
        processor = null;
    }

    if (mediaStreamSource) {
        mediaStreamSource.disconnect();
        mediaStreamSource = null;
    }

    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close();
        audioContext = null;
    }

    // Stop media stream tracks
    if (window._audioStream) {
        window._audioStream.getTracks().forEach(track => track.stop());
        window._audioStream = null;
    }

    // Stop MediaRecorder (legacy)
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        mediaRecorder = null;
    }

    // Send stop message but DON'T close WebSocket yet
    // Wait for server to finish diarization
    if (websocket) {
        const enableDiarization = liveDiarizationCheckbox.checked;
        websocket.send(JSON.stringify({
            type: 'stop',
            enableDiarization
        }));

        // WebSocket will be closed when server sends final status
        // or after timeout
        setTimeout(() => {
            if (websocket) {
                websocket.close();
                websocket = null;
            }
        }, 30000); // 30 second timeout
    }

    isRecording = false;
    startRecordingBtn.disabled = false;
    stopRecordingBtn.disabled = true;

    // Remove streaming indicator from transcription
    if (currentTranscription) {
        transcriptionOutput.innerHTML = `<p>${currentTranscription}</p>`;
    }

    showStatus(recordingStatus, 'Processing complete...', 'info');
}

// Display functions
function clearTranscription() {
    transcriptionOutput.innerHTML = '<p class="placeholder">Transcription will appear here...</p>';
    segmentsOutput.innerHTML = '';
    currentTranscription = '';
}

function displayTranscription(data) {
    currentTranscription = data.full_text;

    // Display full text
    transcriptionOutput.innerHTML = `<p>${data.full_text}</p>`;

    // Display segments with speakers
    if (data.segments && data.segments.length > 0) {
        segmentsOutput.innerHTML = '<h3>Detailed Segments</h3>';

        data.segments.forEach(segment => {
            const segmentDiv = document.createElement('div');
            segmentDiv.className = 'segment';

            const timeStr = `${formatTime(segment.start)} - ${formatTime(segment.end)}`;

            segmentDiv.innerHTML = `
                <div class="segment-header">
                    <div class="segment-info">
                        <span class="segment-speaker">${segment.speaker}</span>
                        <span class="segment-time">${timeStr}</span>
                    </div>
                </div>
                <div class="segment-text">${segment.text}</div>
            `;

            segmentsOutput.appendChild(segmentDiv);
        });
    }

    // Enable LLM processing if connected
    if (llmModelSelect.value) {
        processLLMBtn.disabled = false;
    }
}

function addLiveTranscription(text, start, end, speaker = 'Unknown') {
    currentTranscription += text + ' ';

    // Update full text with streaming indicator
    transcriptionOutput.innerHTML = `
        <div class="streaming-container">
            <span class="streaming-indicator"></span>
            <p>${currentTranscription}</p>
        </div>
    `;

    // Add to segments with animation
    const segmentDiv = document.createElement('div');
    segmentDiv.className = 'live-transcription segment';
    segmentDiv.dataset.start = start;
    segmentDiv.dataset.end = end;

    const timeStr = `${formatTime(start)} - ${formatTime(end)}`;

    segmentDiv.innerHTML = `
        <div class="segment-header">
            <div class="segment-info">
                <span class="segment-speaker">${speaker}</span>
                <span class="segment-time">${timeStr}</span>
            </div>
        </div>
        <div class="segment-text">${text}</div>
    `;

    segmentsOutput.appendChild(segmentDiv);
    segmentsOutput.scrollTop = segmentsOutput.scrollHeight;

    // Enable LLM processing
    if (llmModelSelect.value) {
        processLLMBtn.disabled = false;
    }
}

function updateLiveSegmentsWithSpeakers(segments) {
    // Clear and rebuild segments with speaker information
    segmentsOutput.innerHTML = '<h3>Detailed Segments (with Speaker Diarization)</h3>';

    segments.forEach(segment => {
        const segmentDiv = document.createElement('div');
        segmentDiv.className = 'segment';

        const timeStr = `${formatTime(segment.start)} - ${formatTime(segment.end)}`;

        segmentDiv.innerHTML = `
            <div class="segment-header">
                <div class="segment-info">
                    <span class="segment-speaker">${segment.speaker}</span>
                    <span class="segment-time">${timeStr}</span>
                </div>
            </div>
            <div class="segment-text">${segment.text}</div>
        `;

        segmentsOutput.appendChild(segmentDiv);
    });

    // Update full transcription text
    const fullText = segments.map(s => s.text).join(' ');
    transcriptionOutput.innerHTML = `<p>${fullText}</p>`;
    currentTranscription = fullText;

    showStatus(recordingStatus, 'Speaker diarization complete!', 'success');
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// LLM integration
async function connectToLMStudio() {
    const serverUrl = lmStudioUrl.value;

    try {
        connectLMStudioBtn.disabled = true;
        const response = await fetch('/api/lm-studio/connect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ server_url: serverUrl })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail);
        }

        showStatus(lmStudioStatus, 'Connected to LM Studio', 'success');

        // Load models
        await refreshLLMModels();
    } catch (error) {
        showStatus(lmStudioStatus, `Connection failed: ${error.message}`, 'error');
    } finally {
        connectLMStudioBtn.disabled = false;
    }
}

async function refreshLLMModels() {
    try {
        const response = await fetch('/api/lm-studio/models');
        const data = await response.json();

        // Update model selector
        llmModelSelect.innerHTML = '';

        if (data.models.length === 0) {
            llmModelSelect.innerHTML = '<option value="">No models loaded in LM Studio</option>';
        } else {
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                llmModelSelect.appendChild(option);
            });
        }

        // Enable process button if we have transcription
        if (currentTranscription && llmModelSelect.value) {
            processLLMBtn.disabled = false;
        }
    } catch (error) {
        showStatus(lmStudioStatus, `Error loading models: ${error.message}`, 'error');
    }
}

async function processWithLLM() {
    if (!currentTranscription) {
        alert('No transcription available to process');
        return;
    }

    const model = llmModelSelect.value;
    if (!model) {
        alert('Please select an LLM model');
        return;
    }

    try {
        processLLMBtn.disabled = true;
        llmOutput.style.display = 'block';
        llmResult.innerHTML = '<div class="streaming"><span class="streaming-indicator"></span> Generating summary...</div>';

        const response = await fetch('/api/lm-studio/process/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: currentTranscription,
                model: model,
                system_prompt: systemPrompt.value
            })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        // Handle streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullResult = '';

        // Clear placeholder and create markdown container
        llmResult.innerHTML = '<div class="markdown-content"></div>';
        const contentDiv = llmResult.querySelector('.markdown-content');

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));

                        if (data.type === 'chunk') {
                            fullResult += data.content;
                            // Render markdown in real-time
                            contentDiv.innerHTML = marked.parse(fullResult);
                            // Auto-scroll to bottom
                            llmResult.scrollTop = llmResult.scrollHeight;
                        } else if (data.type === 'error') {
                            throw new Error(data.message);
                        } else if (data.type === 'done') {
                            // Streaming complete - final render
                            contentDiv.innerHTML = marked.parse(fullResult);
                        }
                    } catch (e) {
                        if (e.message && e.message.includes('Server error')) {
                            throw e;
                        }
                        // Ignore JSON parse errors for incomplete lines
                    }
                }
            }
        }

        if (!fullResult) {
            llmResult.innerHTML = '<p class="status-error">No response received from LLM</p>';
        }

    } catch (error) {
        llmResult.innerHTML = `<p class="status-error">Error: ${error.message}</p>`;
    } finally {
        processLLMBtn.disabled = false;
    }
}

// TSE (Target Speaker Extraction) functions
let enrollmentComplete = false;

// Update threshold value display
similarityThreshold.addEventListener('input', () => {
    thresholdValue.textContent = similarityThreshold.value;
});

async function uploadEnrollment() {
    const file = enrollmentAudioInput.files[0];
    if (!file) {
        alert('Please select an enrollment audio file');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        uploadEnrollmentBtn.disabled = true;
        showStatus(enrollmentStatus, 'Uploading enrollment sample...', 'info');

        const response = await fetch('/api/tse/enroll', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail);
        }

        const data = await response.json();
        showStatus(enrollmentStatus, `Enrollment complete! (${data.duration.toFixed(2)}s audio)`, 'success');
        enrollmentComplete = true;
        extractSpeakerBtn.disabled = false;  // Enable extraction button
    } catch (error) {
        showStatus(enrollmentStatus, `Error: ${error.message}`, 'error');
        enrollmentComplete = false;
        extractSpeakerBtn.disabled = true;
    } finally {
        uploadEnrollmentBtn.disabled = false;
    }
}

async function extractSpeaker() {
    if (!enrollmentComplete) {
        alert('Please upload an enrollment sample first');
        return;
    }

    const file = mixedAudioInput.files[0];
    if (!file) {
        alert('Please select a mixed audio file');
        return;
    }

    const model = whisperModelSelect.value;
    const language = languageSelect.value;
    const threshold = parseFloat(similarityThreshold.value);

    const formData = new FormData();
    formData.append('file', file);

    try {
        extractSpeakerBtn.disabled = true;
        clearTranscription();
        showProgress(true, 'Extracting speaker from mixed audio...');
        updateProgress(10);

        const response = await fetch(
            `/api/tse/extract/stream?model=${model}&language=${language}&threshold=${threshold}`,
            {
                method: 'POST',
                body: formData
            }
        );

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        // Handle SSE streaming
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        transcriptionOutput.innerHTML = '<div class="streaming"><span class="streaming-indicator"></span> Extracting speaker...</div>';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.trim() && line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));

                        if (data.type === 'status') {
                            showProgress(true, data.message);
                            updateProgress(30);
                        } else if (data.type === 'progress') {
                            showProgress(true, data.message);
                            updateProgress(50);
                        } else if (data.type === 'warning') {
                            showStatus(extractionStatus, data.message, 'warning');
                        } else if (data.type === 'complete') {
                            // Display results
                            updateProgress(100);

                            // Store transcription
                            currentTranscription = data.full_text;

                            // Display speaker info
                            let speakerInfo = `
                                <div class="tse-info">
                                    <h3>Extraction Results</h3>
                                    <p><strong>Best Match:</strong> ${data.speaker_id} (Similarity: ${(data.similarity * 100).toFixed(1)}%)</p>
                                    <p><strong>Duration:</strong> ${data.duration.toFixed(2)}s</p>
                                    <p><strong>Segments:</strong> ${data.total_segments}</p>
                                </div>
                            `;

                            // Display all speaker similarities
                            if (data.all_speakers && Object.keys(data.all_speakers).length > 1) {
                                speakerInfo += '<div class="speaker-similarities"><h4>All Speakers:</h4><ul>';
                                for (const [speaker, similarity] of Object.entries(data.all_speakers)) {
                                    const percent = (similarity * 100).toFixed(1);
                                    speakerInfo += `<li>${speaker}: ${percent}%</li>`;
                                }
                                speakerInfo += '</ul></div>';
                            }

                            tseResults.innerHTML = speakerInfo;

                            // Display extracted audio player
                            if (data.audio_url) {
                                tseResults.innerHTML += `
                                    <div class="extracted-audio">
                                        <h4>Extracted Audio:</h4>
                                        <audio controls style="width: 100%;">
                                            <source src="${data.audio_url}" type="audio/wav">
                                            Your browser does not support the audio element.
                                        </audio>
                                    </div>
                                `;
                            }

                            // Display transcription
                            displayTranscription(data);

                            showStatus(extractionStatus, 'Speaker extraction complete!', 'success');

                            // Enable LLM processing if available
                            if (llmModelSelect.value) {
                                processLLMBtn.disabled = false;
                            }

                        } else if (data.type === 'error') {
                            throw new Error(data.message);
                        } else if (data.type === 'done') {
                            showProgress(false);
                        }
                    } catch (e) {
                        console.error('Error parsing streaming data:', e);
                    }
                }
            }
        }
    } catch (error) {
        showProgress(false);
        showStatus(extractionStatus, `Error: ${error.message}`, 'error');
        tseResults.innerHTML = '';
    } finally {
        extractSpeakerBtn.disabled = false;
    }
}

// Event listeners
refreshWhisperBtn.addEventListener('click', refreshWhisperModels);
unloadModelBtn.addEventListener('click', unloadWhisperModel);
unloadAllBtn.addEventListener('click', unloadAll);

transcribeFileBtn.addEventListener('click', transcribeFile);

startRecordingBtn.addEventListener('click', startLiveRecording);
stopRecordingBtn.addEventListener('click', stopLiveRecording);

connectLMStudioBtn.addEventListener('click', connectToLMStudio);
refreshLLMBtn.addEventListener('click', refreshLLMModels);
processLLMBtn.addEventListener('click', processWithLLM);

uploadEnrollmentBtn.addEventListener('click', uploadEnrollment);
extractSpeakerBtn.addEventListener('click', extractSpeaker);

// Initialize
refreshWhisperModels();
