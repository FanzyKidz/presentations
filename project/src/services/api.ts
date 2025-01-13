import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // Update with your Python API URL

export const sendMessage = async (content: string, files: File[]) => {
  const formData = new FormData();
  formData.append('message', content);
  files.forEach(file => formData.append('files', file));

  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    body: formData,
  });

  const reader = response.body?.getReader();
  if (!reader) throw new Error('No reader available');

  return {
    async *[Symbol.asyncIterator]() {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const text = new TextDecoder().decode(value);
        const chunks = text.split('\n').filter(Boolean);
        
        for (const chunk of chunks) {
          yield chunk;
        }
      }
    }
  };
};