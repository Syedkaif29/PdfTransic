const API_BASE_URL = 'http://localhost:9001';

export interface TranslationRequest {
  text: string;
  target_language: string;
  source_language?: string;
}

export interface TranslationResponse {
  translated_text: string;
  original_text: string;
  source_language: string;
  target_language: string;
  success: boolean;
}

export interface ApiError {
  detail: string;
}

export class TranslationApiService {
  static async translateText(request: TranslationRequest): Promise<TranslationResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/translate-simple`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: request.text,
          target_language: request.target_language,
          source_language: request.source_language || 'eng_Latn'
        }),
      });

      if (!response.ok) {
        const errorData: ApiError = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data: TranslationResponse = await response.json();
      return data;
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(`Translation failed: ${error.message}`);
      }
      throw new Error('Translation failed: Unknown error');
    }
  }

  static async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const data = await response.json();
      return data.status === 'healthy';
    } catch (error) {
      return false;
    }
  }

  static async getSupportedLanguages() {
    try {
      const response = await fetch(`${API_BASE_URL}/languages`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Failed to fetch supported languages:', error);
      return null;
    }
  }
}