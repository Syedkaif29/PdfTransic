import { useState } from 'react';
import type { ChangeEvent, FormEvent } from 'react';

interface TranslationFormProps {
  onSubmit: (data: FormData) => void;
  isLoading?: boolean;
}

interface FormData {
  text: string;
  file: File | null;
  targetLanguage: string;
}

const indianLanguages = [
  { code: 'hi', name: 'Hindi' },
  { code: 'ta', name: 'Tamil' },
  { code: 'te', name: 'Telugu' },
  { code: 'bn', name: 'Bengali' },
  { code: 'gu', name: 'Gujarati' },
  { code: 'mr', name: 'Marathi' },
  { code: 'ml', name: 'Malayalam' },
  { code: 'pa', name: 'Punjabi' },
  { code: 'kn', name: 'Kannada' },
  { code: 'or', name: 'Odia' }
];

export const TranslationForm = ({ onSubmit, isLoading = false }: TranslationFormProps) => {
  const [formData, setFormData] = useState<FormData>({
    text: '',
    file: null,
    targetLanguage: indianLanguages[0].code
  });

  const handleTextChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setFormData(prev => ({ ...prev, text: e.target.value }));
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setFormData(prev => ({ ...prev, file }));
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
        Indian Language Translator
      </h2>
      <p className="text-center text-gray-500 text-sm">
        Translate text or documents into your preferred Indian language.
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

      {/* File and Language in a Row */}
      <div className="flex flex-col md:flex-row gap-4">
        {/* File Upload */}
        <div className="flex-1">
          <label htmlFor="file" className="block text-sm font-medium text-gray-700 mb-1">
            Or Upload a Document (PDF/DOCX)
          </label>
          <input
            type="file"
            id="file"
            accept=".pdf,.doc,.docx"
            onChange={handleFileChange}
            className="mt-1 block w-full text-sm text-gray-500 
              file:mr-4 file:py-2 file:px-4 
              file:rounded-lg file:border-0 
              file:text-sm file:font-semibold 
              file:bg-indigo-50 file:text-indigo-700 
              hover:file:bg-indigo-100"
          />
        </div>

        {/* Language Selector */}
        <div className="flex-1">
          <label htmlFor="language" className="block text-sm font-medium text-gray-700 mb-1">
            Select Target Language
          </label>
          <select
            id="language"
            className="mt-1 block w-full rounded-lg border border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 focus:ring-offset-1 p-2 bg-white text-gray-900"
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
