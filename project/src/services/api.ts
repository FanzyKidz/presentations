import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const sendMessage = async (content: string, files: File[]) => {
  try {
    const formData = new FormData();
    formData.append('message', content.trim());
    
    if (files && files.length > 0) {
      files.forEach(file => formData.append('files', file));
    }

    console.log('Sending request:', {
      content: content.trim(),
      filesCount: files?.length || 0
    });

    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Server error:', {
        status: response.status,
        statusText: response.statusText,
        error: errorText
      });
      throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
    }

    const data = await response.json();
    return data.text;
  } catch (error) {
    console.error('API error:', error);
    throw error;
  }
};
