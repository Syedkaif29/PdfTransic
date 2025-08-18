import { useState } from 'react'
import { TranslationForm } from './components/TranslationForm'

interface FormData {
  text: string;
  file: File | null;
  targetLanguage: string;
}

export default function App() {
  const [translatedText, setTranslatedText] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);

  const handleTranslation = async (data: FormData) => {
    setIsLoading(true);
    // TODO: Implement API call to FastAPI backend
    console.log('Translation requested:', data);
    await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API call
    setTranslatedText('Translation will appear here...');
    setIsLoading(false);
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
        </div>

        {/* Main Content */}
        <div className="w-full max-w-6xl mx-auto">
          <div className="bg-white/80 backdrop-blur-xl rounded-2xl shadow-2xl border border-indigo-50 overflow-hidden">
            <div className="p-8">
              <TranslationForm onSubmit={handleTranslation} isLoading={isLoading} />
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
