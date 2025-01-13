import React, { useState, useRef, useEffect } from 'react';
import { Message } from './types/chat';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { sendMessage } from './services/api';
import { Bot } from 'lucide-react';

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (content: string, files: File[]) => {
    if (!content.trim() && files.length === 0) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      role: 'user',
      files,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const stream = await sendMessage(content, files);
      let assistantMessage = '';

      const assistantMessageId = Date.now().toString();
      for await (const chunk of stream) {
        assistantMessage += chunk;
        setMessages(prev => {
          const existing = prev.find(m => m.id === assistantMessageId);
          if (existing) {
            return prev.map(m =>
              m.id === assistantMessageId
                ? { ...m, content: assistantMessage }
                : m
            );
          }
          return [
            ...prev,
            {
              id: assistantMessageId,
              content: assistantMessage,
              role: 'assistant',
              timestamp: new Date(),
            },
          ];
        });
      }
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 px-4 py-3">
        <div className="flex items-center gap-2">
          <Bot className="w-6 h-6 text-blue-500" />
          <h1 className="text-xl font-semibold">AI Assistant</h1>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map(message => (
          <ChatMessage key={message.id} message={message} />
        ))}
        <div ref={messagesEndRef} />
      </main>

      <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
    </div>
  );
}

export default App;