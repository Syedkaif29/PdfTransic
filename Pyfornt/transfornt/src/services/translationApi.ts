import { API_CONFIG, getApiUrl } from '../config/api';

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

export interface PdfTranslationResponse {
  success: boolean;
  filename: string;
  pages_processed: number;
  extracted_text: string;
  translated_text: string;
  target_language: string;
  source_language: string;
}

export class TranslationApiService {
  static async translateText(request: TranslationRequest): Promise<TranslationResponse> {
    try {
      const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.TRANSLATE_SIMPLE), {
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
      const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.HEALTH));
      const data = await response.json();
      return data.status === 'healthy';
    } catch (error) {
      return false;
    }
  }

  static async getSupportedLanguages() {
    try {
      const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.LANGUAGES));
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Failed to fetch supported languages:', error);
      return null;
    }
  }

  static async translatePdf(file: File, targetLanguage: string): Promise<PdfTranslationResponse> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('target_language', targetLanguage);

      const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.TRANSLATE_PDF), {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData: ApiError = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data: PdfTranslationResponse = await response.json();
      return data;
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(`PDF translation failed: ${error.message}`);
      }
      throw new Error('PDF translation failed: Unknown error');
    }
  }
}