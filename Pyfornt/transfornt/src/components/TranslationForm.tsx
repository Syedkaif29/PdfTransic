import { useState } from 'react';
import type { ChangeEvent, FormEvent } from 'react';

interface TranslationFormProps {
  onSubmit: (data: FormData) => void;
  isLoading?: boolean;
}

interface FormData {
  text: string;
  targetLanguage: string;
}

const indianLanguages = [
  { code: 'asm_Beng', name: 'Assamese' },
  { code: 'ben_Beng', name: 'Bengali' },
  { code: 'brx_Deva', name: 'Bodo' },
  { code: 'doi_Deva', name: 'Dogri' },
  { code: 'gom_Deva', name: 'Konkani' },
  { code: 'guj_Gujr', name: 'Gujarati' },
  { code: 'hin_Deva', name: 'Hindi' },
  { code: 'kan_Knda', name: 'Kannada' },
  { code: 'kas_Arab', name: 'Kashmiri (Arabic)' },
  { code: 'kas_Deva', name: 'Kashmiri (Devanagari)' },
  { code: 'mai_Deva', name: 'Maithili' },
  { code: 'mal_Mlym', name: 'Malayalam' },
  { code: 'mni_Beng', name: 'Manipuri (Bengali)' },
  { code: 'mni_Mtei', name: 'Manipuri (Meitei)' },
  { code: 'mar_Deva', name: 'Marathi' },
  { code: 'npi_Deva', name: 'Nepali' },
  { code: 'ory_Orya', name: 'Odia' },
  { code: 'pan_Guru', name: 'Punjabi' },
  { code: 'san_Deva', name: 'Sanskrit' },
  { code: 'sat_Olck', name: 'Santali' },
  { code: 'snd_Arab', name: 'Sindhi (Arabic)' },
  { code: 'snd_Deva', name: 'Sindhi (Devanagari)' },
  { code: 'tam_Taml', name: 'Tamil' },
  { code: 'tel_Telu', name: 'Telugu' },
  { code: 'urd_Arab', name: 'Urdu' }
];

export const TranslationForm = ({ onSubmit, isLoading = false }: TranslationFormProps) => {
  const [formData, setFormData] = useState<FormData>({
    text: '',
    targetLanguage: indianLanguages[0].code
  });

  const handleTextChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setFormData(prev => ({ ...prev, text: e.target.value }));
  };



  const handleLanguageChange = (e: ChangeEvent<HTMLSelectElement>) => {
    setFormData(prev => ({ ...prev, targetLanguage: e.target.value }));
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="max-w-2xl mx-auto p-8 rounded-2xl shadow-xl bg-white/80 backdrop-blur-md border border-gray-200 space-y-6"
    >
      <h2 className="text-2xl font-bold text-gray-800 text-center">
        Text Translation
      </h2>
      <p className="text-center text-gray-500 text-sm">
        Enter text to translate into your preferred Indian language.
      </p>

      {/* Textarea */}
      <div>
        <label htmlFor="text" className="block text-sm font-medium text-gray-700 mb-1">
          Enter Text to Translate
        </label>
        <textarea
          id="text"
          rows={5}
          className="mt-1 block w-full rounded-lg border border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 focus:ring-offset-1 p-3 resize-none bg-white text-gray-900 placeholder-gray-500"
          placeholder="Type or paste your text here..."
          value={formData.text}
          onChange={handleTextChange}
          style={{
            color: '#111827',
            backgroundColor: '#ffffff'
          }}
        />
      </div>

      {/* Language Selector */}
      <div>
        <label htmlFor="language" className="block text-sm font-medium text-gray-700 mb-1">
          Select Target Language
        </label>
        <select
          id="language"
          className="mt-1 block w-full rounded-lg border border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 focus:ring-offset-1 p-3 bg-white text-gray-900"
          value={formData.targetLanguage}
          onChange={handleLanguageChange}
          style={{
            color: '#111827',
            backgroundColor: '#ffffff'
          }}
        >
          {indianLanguages.map(lang => (
            <option 
              key={lang.code} 
              value={lang.code}
              className="bg-white text-gray-900 hover:bg-indigo-50"
              style={{
                backgroundColor: '#ffffff',
                color: '#111827',
                padding: '8px'
              }}
            >
              {lang.name}
            </option>
          ))}
        </select>
      </div>

      {/* Button */}
      <button
        type="submit"
        className="w-full py-3 px-6 rounded-lg text-white font-medium text-lg bg-gradient-to-r from-indigo-500 to-purple-500 hover:from-indigo-600 hover:to-purple-600 focus:ring-4 focus:ring-indigo-300 disabled:opacity-60 disabled:cursor-not-allowed transition-all"
        disabled={isLoading}
      >
        {isLoading ? "Translating..." : "Translate Now"}
      </button>
    </form>
  );
};
