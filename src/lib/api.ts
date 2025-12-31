// Get backend URL from environment variable or default to localhost for development
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
const API_BASE = '/api';

interface ApiError {
  error: string;
}

interface User {
  id: number;
  name: string;
  email: string;
}

interface AuthResponse {
  id: number;
  name: string;
  email: string;
}

async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error((data as ApiError).error || 'Request failed');
  }

  return data as T;
}

export const api = {
  register: async (name: string, email: string, password: string): Promise<User> => {
    return fetchApi<AuthResponse>('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ name, email, password }),
    });
  },

  login: async (email: string, password: string): Promise<User> => {
    return fetchApi<AuthResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  },

  logout: async (): Promise<void> => {
    await fetchApi<{ ok: boolean }>('/auth/logout', {
      method: 'POST',
    });
  },

  me: async (): Promise<User> => {
    return fetchApi<AuthResponse>('/auth/me');
  },
};

// Video prediction types and API
export interface VideoPredictionRequest {
  file: File;
  frameSamplingRate: number;
  faceFocus: boolean;
}

export interface VideoPredictionResponse {
  prediction: 'REAL' | 'FAKE';
  confidence: number;
  preprocessedImages?: string[];
  faceCroppedImages?: string[];
  error?: string;
}

export const predictVideo = async (
  request: VideoPredictionRequest
): Promise<VideoPredictionResponse> => {
  const formData = new FormData();
  formData.append('file', request.file);
  formData.append('sequence_length', request.frameSamplingRate.toString());
  formData.append('face_focus', request.faceFocus.toString());

  try {
    const response = await fetch(`${BACKEND_URL}/api/predict`, {
      method: 'POST',
      body: formData,
      // Don't set Content-Type header - browser will set it with boundary for multipart/form-data
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Request failed' }));
      throw new Error(errorData.error || `Server error: ${response.status}`);
    }

    const data = await response.json();
    return data as VideoPredictionResponse;
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Failed to connect to prediction server');
  }
};
