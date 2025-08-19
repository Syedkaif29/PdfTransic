// API Configuration
export const API_CONFIG = {
  // Get base URL from environment variables
  BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  
  // API endpoints
  ENDPOINTS: {
    TRANSLATE_SIMPLE: '/translate-simple',
    TRANSLATE_PDF: '/translate-pdf',
    HEALTH: '/health',
    LANGUAGES: '/languages',
    TRANSLATE: '/translate'
  },
  
  // Request timeouts
  TIMEOUTS: {
    DEFAULT: 30000, // 30 seconds
    TRANSLATION: 60000, // 60 seconds for translations
    PDF_UPLOAD: 120000 // 2 minutes for PDF processing
  }
};

// Helper function to get full endpoint URL
export const getApiUrl = (endpoint: string): string => {
  return `${API_CONFIG.BASE_URL}${endpoint}`;
};

// Environment info for debugging
export const ENV_INFO = {
  isDevelopment: import.meta.env.DEV,
  isProduction: import.meta.env.PROD,
  apiBaseUrl: API_CONFIG.BASE_URL,
  mode: import.meta.env.MODE
};