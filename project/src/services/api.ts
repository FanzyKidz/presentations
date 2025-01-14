import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const sendMessage = async (content: string, files: File[]) => {
  try {
    const formData = new FormData();
    formData.append('message', content.trim());
    
    if (files && files.length > 0) {
      files.forEach(file => formData.append('files', file));
    }

    console.log('Sending message:', content); // Debug log

    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Server error:', errorText); // Debug log
      throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No reader available');

    return {
      async *[Symbol.asyncIterator]() {
        const decoder = new TextDecoder();
        let buffer = '';

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6).trim();
                if (data === '[DONE]') return;
                
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
        } catch (error) {
          console.error('Stream error:', error);
          throw error;
        }
      }
    };
  } catch (error) {
    console.error('API error:', error);
    throw error;
  }
};
