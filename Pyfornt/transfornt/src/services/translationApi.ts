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
  layout_data_available?: boolean; // New field
  original_text_chunks: string[]; // New field
  translated_text_chunks: string[]; // New field
  layout_data?: any[]; // New field for PyMuPDF layout data
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
    originalTextChunks: string[], // Changed from originalText: string
    translatedTextChunks: string[], // Changed from translatedText: string
    filename: string,
    targetLanguage: string,
    layoutData: any[] // New parameter for layout data
  ): Promise<void> {
    try {
      const formData = new FormData();
      formData.append('original_text_chunks_json', JSON.stringify(originalTextChunks)); // Stringify
      formData.append('translated_text_chunks_json', JSON.stringify(translatedTextChunks)); // Stringify
      formData.append('layout_data_json', JSON.stringify(layoutData)); // Stringify
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

  static streamLivePreview(file: File, targetLanguage: string, onEvent: (event: any) => void, onError?: (error: Error) => void, onComplete?: () => void) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_language', targetLanguage);

    // We need to manually construct the request for SSE (can't use fetch with FormData for SSE)
    const xhr = new XMLHttpRequest();
    xhr.open('POST', getApiUrl('/translate-pdf-live-preview'), true);
    xhr.responseType = 'text';

    xhr.onreadystatechange = function () {
      if (xhr.readyState === XMLHttpRequest.DONE) {
        if (xhr.status !== 200 && onError) {
          onError(new Error(`Live preview failed: ${xhr.statusText}`));
        } else if (onComplete) {
          onComplete();
        }
      }
    };

    let lastIndex = 0;
    xhr.onprogress = function () {
      const newText = xhr.responseText.substring(lastIndex);
      lastIndex = xhr.responseText.length;
      // Split by double newlines (SSE event separator)
      const events = newText.split(/\n\n+/);
      for (const event of events) {
        if (event.startsWith('data: ')) {
          try {
            const json = JSON.parse(event.replace('data: ', ''));
            onEvent(json);
          } catch (e) {
            // Ignore parse errors
          }
        }
      }
    };

    xhr.onerror = function () {
      if (onError) onError(new Error('Live preview connection error'));
    };

    xhr.send(formData);
    return xhr; // Allow caller to abort if needed
  }
}