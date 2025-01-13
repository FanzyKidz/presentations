export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  files?: File[];
  timestamp: Date;
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
}