import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

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
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        // Keep the last partial line in the buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') break;
            
            try {
              const parsed = JSON.parse(data);
              if (parsed.text) {
                yield parsed.text;
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e);
            }
          }
        }
      }
    }
  };
};
