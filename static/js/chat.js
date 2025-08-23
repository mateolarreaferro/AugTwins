class ChatInterface {
    constructor() {
        this.chatContainer = document.getElementById('chatContainer');
        this.messageInput = document.getElementById('messageInput');
        this.currentAgentSpan = document.getElementById('currentAgent');
        this.placeholder = this.chatContainer.querySelector('.chat-placeholder');
        
        this.audioContext = null;
        this.websocket = null;
        this.isPlayingAudio = false;
        
        this.initializeEventListeners();
        this.initializeAudio();
    }
    
    initializeEventListeners() {
        // Focus on input when page loads
        this.messageInput.focus();
        
        // Auto-resize input based on content
        this.messageInput.addEventListener('input', this.autoResizeInput.bind(this));
    }
    
    async initializeAudio() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 22050
            });
        } catch (error) {
            console.warn('Audio not available:', error);
        }
    }
    
    connectWebSocket() {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            return this.websocket;
        }
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.websocket = new WebSocket(wsUrl);
        this.websocket.binaryType = 'arraybuffer';
        
        this.websocket.onopen = () => {
            console.log('[Debug TTS] WebSocket connected');
        };
        
        this.websocket.onerror = (error) => {
            console.error('[Debug TTS] WebSocket error:', error);
        };
        
        this.websocket.onclose = () => {
            console.log('[Debug TTS] WebSocket closed');
            this.websocket = null;
        };
        
        return this.websocket;
    }
    
    async playTTSAudio(text) {
        if (!this.audioContext || this.isPlayingAudio) {
            console.log('[Debug TTS] Audio context unavailable or already playing');
            return;
        }
        
        try {
            // Resume audio context if suspended (browser autoplay policy)
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }
            
            const ws = this.connectWebSocket();
            if (ws.readyState !== WebSocket.OPEN) {
                // Wait for connection
                await new Promise((resolve, reject) => {
                    const timeout = setTimeout(() => reject(new Error('WebSocket connection timeout')), 5000);
                    ws.onopen = () => {
                        clearTimeout(timeout);
                        resolve();
                    };
                    ws.onerror = () => {
                        clearTimeout(timeout);
                        reject(new Error('WebSocket connection failed'));
                    };
                });
            }
            
            this.isPlayingAudio = true;
            this.addSystemMessage('🔊 Playing audio response...');
            
            // Send TTS request
            const jobId = `debug_${Date.now()}`;
            ws.send(JSON.stringify({
                type: 'prompt',
                text: text,
                id: jobId
            }));
            
            // Handle WebSocket messages
            const audioChunks = [];
            let audioStarted = false;
            
            const handleMessage = async (event) => {
                if (event.data instanceof ArrayBuffer) {
                    // Binary PCM audio data
                    if (audioStarted) {
                        audioChunks.push(new Uint8Array(event.data));
                    }
                } else {
                    // JSON message
                    try {
                        const message = JSON.parse(event.data);
                        
                        if (message.type === 'audio_start' && message.id === jobId) {
                            console.log('[Debug TTS] Audio stream started');
                            audioStarted = true;
                        } else if (message.type === 'audio_end' && message.id === jobId) {
                            console.log('[Debug TTS] Audio stream ended, playing audio');
                            ws.removeEventListener('message', handleMessage);
                            
                            // Convert and play audio
                            if (audioChunks.length > 0) {
                                await this.playPCMAudio(audioChunks);
                            }
                            
                            this.isPlayingAudio = false;
                        } else if (message.type === 'error') {
                            console.error('[Debug TTS] Error:', message.error);
                            ws.removeEventListener('message', handleMessage);
                            this.addSystemMessage(`🔊 Audio error: ${message.error}`);
                            this.isPlayingAudio = false;
                        }
                    } catch (e) {
                        console.error('[Debug TTS] Failed to parse message:', e);
                    }
                }
            };
            
            ws.addEventListener('message', handleMessage);
            
        } catch (error) {
            console.error('[Debug TTS] Failed to play audio:', error);
            this.addSystemMessage(`🔊 Audio failed: ${error.message}`);
            this.isPlayingAudio = false;
        }
    }
    
    async playPCMAudio(chunks) {
        try {
            // Combine all chunks
            const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
            const combinedData = new Uint8Array(totalLength);
            let offset = 0;
            for (const chunk of chunks) {
                combinedData.set(chunk, offset);
                offset += chunk.length;
            }
            
            // Convert PCM S16LE to Float32Array
            const samples = new Float32Array(combinedData.length / 2);
            const dataView = new DataView(combinedData.buffer);
            
            for (let i = 0; i < samples.length; i++) {
                // Read 16-bit signed integer (little-endian) and convert to float
                const int16 = dataView.getInt16(i * 2, true);
                samples[i] = int16 / 32768.0; // Convert to -1.0 to 1.0 range
            }
            
            // Create audio buffer
            const audioBuffer = this.audioContext.createBuffer(1, samples.length, 22050);
            audioBuffer.getChannelData(0).set(samples);
            
            // Play audio
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            source.start();
            
            console.log(`[Debug TTS] Playing ${samples.length} samples (${(samples.length / 22050).toFixed(2)}s)`);
            
        } catch (error) {
            console.error('[Debug TTS] Failed to play PCM audio:', error);
            throw error;
        }
    }
    
    autoResizeInput() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }
    
    hidePlaceholder() {
        if (this.placeholder && this.placeholder.style.display !== 'none') {
            this.placeholder.style.display = 'none';
        }
    }
    
    addMessage(sender, message, isUser = false) {
        this.hidePlaceholder();
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'agent-message'}`;
        
        const senderSpan = document.createElement('div');
        senderSpan.className = 'message-sender';
        senderSpan.textContent = sender;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = message;
        
        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'message-timestamp';
        timestampDiv.textContent = new Date().toLocaleTimeString();
        
        messageDiv.appendChild(senderSpan);
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timestampDiv);
        
        this.chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Add animation
        messageDiv.style.opacity = '0';
        messageDiv.style.transform = 'translateY(20px)';
        setTimeout(() => {
            messageDiv.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            messageDiv.style.opacity = '1';
            messageDiv.style.transform = 'translateY(0)';
        }, 10);
    }
    
    scrollToBottom() {
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }
    
    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message agent-message typing-indicator';
        typingDiv.id = 'typing-indicator';
        
        const dotsDiv = document.createElement('div');
        dotsDiv.className = 'typing-dots';
        dotsDiv.innerHTML = '<span></span><span></span><span></span>';
        
        typingDiv.appendChild(dotsDiv);
        this.chatContainer.appendChild(typingDiv);
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;
        
        this.addMessage('You', message, true);
        this.messageInput.value = '';
        this.autoResizeInput();
        this.messageInput.disabled = true;
        
        this.showTypingIndicator();
        
        try {
            const mode = document.getElementById('modeSelect').value;
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    message: message,
                    mode: mode 
                })
            });
            
            const data = await response.json();
            this.hideTypingIndicator();
            
            if (response.ok) {
                this.addMessage(data.agent, data.response);
                
                // Use WebSocket-based TTS for lower latency
                if (data.response && this.audioContext) {
                    this.playTTSAudio(data.response);
                }
            } else {
                this.addSystemMessage(`Error: ${data.error || 'Unknown error'}`);
            }
        } catch (error) {
            this.hideTypingIndicator();
            this.addSystemMessage(`Connection error: ${error.message}`);
        } finally {
            this.messageInput.disabled = false;
            this.messageInput.focus();
        }
    }
    
    addSystemMessage(message) {
        this.addMessage('System', message);
    }
    
    async switchAgent() {
        const selectedAgent = document.getElementById('agentSelect').value;
        const switchButton = document.querySelector('.btn-secondary');
        const originalText = switchButton.textContent;
        
        switchButton.textContent = 'Switching...';
        switchButton.disabled = true;
        
        try {
            const response = await fetch('/switch-agent', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ agent: selectedAgent })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.currentAgentSpan.textContent = data.current_agent;
                this.addSystemMessage(`Switched to ${data.current_agent}`);
            } else {
                this.addSystemMessage(`Error switching agent: ${data.error}`);
            }
        } catch (error) {
            this.addSystemMessage(`Error switching agent: ${error.message}`);
        } finally {
            switchButton.textContent = originalText;
            switchButton.disabled = false;
        }
    }
    
    async saveConversation() {
        const saveButton = document.querySelector('.btn-primary');
        const originalText = saveButton.textContent;
        
        saveButton.textContent = 'Saving...';
        saveButton.disabled = true;
        
        try {
            const response = await fetch('/save-conversation', { method: 'POST' });
            const data = await response.json();
            
            if (response.ok) {
                this.addSystemMessage(data.message);
            } else {
                this.addSystemMessage(`Error: ${data.error}`);
            }
        } catch (error) {
            this.addSystemMessage(`Error saving conversation: ${error.message}`);
        } finally {
            saveButton.textContent = originalText;
            saveButton.disabled = false;
        }
    }
    
    handleKeyPress(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
    }
}

// Initialize the chat interface
const chatInterface = new ChatInterface();

// Global functions for onclick handlers
function sendMessage() {
    chatInterface.sendMessage();
}

function switchAgent() {
    chatInterface.switchAgent();
}

function saveConversation() {
    chatInterface.saveConversation();
}

function handleKeyPress(event) {
    chatInterface.handleKeyPress(event);
}