import { useState, useEffect } from 'react'
import { TranslationForm } from './components/TranslationForm'
import { TranslationApiService } from './services/translationApi'

interface FormData {
  text: string;
  file: File | null;
  targetLanguage: string;
}

export default function App() {
  const [translatedText, setTranslatedText] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);

  // Check backend health on component mount
  useEffect(() => {
    const checkBackendHealth = async () => {
      const healthy = await TranslationApiService.checkHealth();
      setIsBackendHealthy(healthy);
    };
    
    checkBackendHealth();
    // Check health every 30 seconds
    const interval = setInterval(checkBackendHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleTranslation = async (data: FormData) => {
    if (!data.text.trim()) {
      setError('Please enter some text to translate');
      return;
    }

    setIsLoading(true);
    setError('');
    setTranslatedText('');

    try {
      const response = await TranslationApiService.translateText({
        text: data.text,
        target_language: data.targetLanguage,
        source_language: 'eng_Latn'
      });

      setTranslatedText(response.translated_text);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Translation failed';
      setError(errorMessage);
      console.error('Translation error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-100 via-white to-purple-100">
      <div className="w-full px-4 py-12">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl md:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600 mb-4">
            Indian Language Translator
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Translate text and documents into various Indian languages instantly
          </p>
          
          {/* Backend Status Indicator */}
          <div className="mt-4 flex justify-center">
            <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
              isBackendHealthy === null 
                ? 'bg-gray-100 text-gray-600' 
                : isBackendHealthy 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-red-100 text-red-800'
            }`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${
                isBackendHealthy === null 
                  ? 'bg-gray-400' 
                  : isBackendHealthy 
                    ? 'bg-green-500' 
                    : 'bg-red-500'
              }`}></div>
              {isBackendHealthy === null 
                ? 'Checking backend...' 
                : isBackendHealthy 
                  ? 'Backend ready' 
                  : 'Backend unavailable'
              }
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="w-full max-w-6xl mx-auto">
          <div className="bg-white/80 backdrop-blur-xl rounded-2xl shadow-2xl border border-indigo-50 overflow-hidden">
            <div className="p-8">
              <TranslationForm onSubmit={handleTranslation} isLoading={isLoading} />
              
              {/* Error Display */}
              {error && (
                <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <div className="flex items-center">
                    <svg className="w-5 h-5 text-red-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <p className="text-red-700 text-sm font-medium">{error}</p>
                  </div>
                </div>
              )}
            </div>
          </div>
          
          {/* Translation Result */}
          {translatedText && (
            <div className="mt-8 bg-white/80 backdrop-blur-xl rounded-2xl shadow-2xl border border-purple-50 overflow-hidden transition-all duration-500 ease-in-out">
              <div className="p-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
                  <svg className="w-6 h-6 mr-2 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                  Translation Result
                </h2>
                <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-6 text-gray-800 whitespace-pre-wrap shadow-inner">
                  {translatedText}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
