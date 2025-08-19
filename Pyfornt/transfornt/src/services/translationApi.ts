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
  extraction_method?: string;
  text_length?: number;
  chunks_processed?: number;
  memory_management?: string;
  duplicates_removed?: boolean;
  download_available?: boolean;
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

  static async translatePdfEnhanced(file: File, targetLanguage: string): Promise<PdfTranslationResponse> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('target_language', targetLanguage);

      const response = await fetch(getApiUrl('/translate-pdf-enhanced'), {
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
        throw new Error(`Enhanced PDF translation failed: ${error.message}`);
      }
      throw new Error('Enhanced PDF translation failed: Unknown error');
    }
  }

  static async downloadTranslatedPdf(
    originalText: string,
    translatedText: string,
    filename: string,
    targetLanguage: string
  ): Promise<void> {
    try {
      const formData = new FormData();
      formData.append('original_text', originalText);
      formData.append('translated_text', translatedText);
      formData.append('filename', filename);
      formData.append('target_language', targetLanguage);

      const response = await fetch(getApiUrl('/download-translated-pdf'), {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData: ApiError = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      // Handle the PDF download
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      // Extract filename from response headers or create one
      const contentDisposition = response.headers.get('Content-Disposition');
      let downloadFilename = `${filename.replace('.pdf', '')}_translated_${targetLanguage}.pdf`;
      
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/);
        if (filenameMatch) {
          downloadFilename = filenameMatch[1];
        }
      }
      
      link.download = downloadFilename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(`PDF download failed: ${error.message}`);
      }
      throw new Error('PDF download failed: Unknown error');
    }
  }

  static async clearMemory(): Promise<{ status: string; message: string }> {
    try {
      const response = await fetch(getApiUrl('/clear-memory'), {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(`Memory clear failed: ${error.message}`);
      }
      throw new Error('Memory clear failed: Unknown error');
    }
  }

  static async getMemoryInfo(): Promise<any> {
    try {
      const response = await fetch(getApiUrl('/memory-info'));
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(`Memory info failed: ${error.message}`);
      }
      throw new Error('Memory info failed: Unknown error');
    }
  }
}