import React, { useState } from 'react';
import { MessageCircle, X, Send } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import api from '../api';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
}

export const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    { id: '1', text: 'Hello! I am your Bookstore AI assistant. How can I help you today?', sender: 'bot' }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const { user } = useAuth();

  const toggleChat = () => setIsOpen(!isOpen);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg: Message = { id: Date.now().toString(), text: input, sender: 'user' };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    try {
      // Assuming the chatbot service expects { user_id, message } or similar
      const res = await api.post('/chatbot/', {
        user_id: user?.id || 'anonymous',
        message: userMsg.text
      });
      
      const botMsg: Message = {
        id: (Date.now() + 1).toString(),
        text: res.data.response || res.data.answer || 'Sorry, I could not process your request.',
        sender: 'bot'
      };
      setMessages(prev => [...prev, botMsg]);
    } catch (error) {
      console.error('Chatbot error:', error);
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        text: 'Sorry, I am currently unavailable. Please try again later.',
        sender: 'bot'
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="chatbot-container">
      {isOpen && (
        <div className="chatbot-window">
          <div className="chatbot-header">
            <h3>AI Assistant</h3>
            <button onClick={toggleChat} className="btn-icon"><X size={20} /></button>
          </div>
          <div className="chatbot-messages">
            {messages.map(msg => (
              <div key={msg.id} className={`message ${msg.sender}`}>
                <div className="message-bubble">{msg.text}</div>
              </div>
            ))}
            {isTyping && (
              <div className="message bot">
                <div className="message-bubble typing">...</div>
              </div>
            )}
          </div>
          <form className="chatbot-input" onSubmit={handleSend}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask for recommendations..."
            />
            <button type="submit" className="btn-icon" disabled={!input.trim()}><Send size={20} /></button>
          </form>
        </div>
      )}
      {!isOpen && (
        <button className="chatbot-toggle" onClick={toggleChat}>
          <MessageCircle size={28} />
        </button>
      )}
    </div>
  );
};
