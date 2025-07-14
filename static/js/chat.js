class ChatInterface {
    constructor() {
        this.chatContainer = document.getElementById('chatContainer');
        this.messageInput = document.getElementById('messageInput');
        this.currentAgentSpan = document.getElementById('currentAgent');
        this.placeholder = this.chatContainer.querySelector('.chat-placeholder');
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // Focus on input when page loads
        this.messageInput.focus();
        
        // Auto-resize input based on content
        this.messageInput.addEventListener('input', this.autoResizeInput.bind(this));
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
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message })
            });
            
            const data = await response.json();
            this.hideTypingIndicator();
            
            if (response.ok) {
                this.addMessage(data.agent, data.response);
                
                if (data.audio_enabled) {
                    this.addSystemMessage('Playing audio response...');
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