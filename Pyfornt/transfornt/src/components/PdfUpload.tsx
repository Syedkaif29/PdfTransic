import React, { useState, useRef } from 'react';
import { TranslationApiService, type PdfTranslationResponse } from '../services/translationApi';

interface PdfUploadProps {
  onTranslationComplete?: (result: PdfTranslationResponse) => void;
}

const PdfUpload: React.FC<PdfUploadProps> = ({ onTranslationComplete }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [targetLanguage, setTargetLanguage] = useState('hin_Deva');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PdfTranslationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const languages = {
    'asm_Beng': 'Assamese',
    'ben_Beng': 'Bengali',
    'brx_Deva': 'Bodo',
    'doi_Deva': 'Dogri',
    'gom_Deva': 'Konkani',
    'guj_Gujr': 'Gujarati',
    'hin_Deva': 'Hindi',
    'kan_Knda': 'Kannada',
    'kas_Arab': 'Kashmiri (Arabic)',
    'kas_Deva': 'Kashmiri (Devanagari)',
    'mai_Deva': 'Maithili',
    'mal_Mlym': 'Malayalam',
    'mni_Beng': 'Manipuri (Bengali)',
    'mni_Mtei': 'Manipuri (Meitei)',
    'mar_Deva': 'Marathi',
    'npi_Deva': 'Nepali',
    'ory_Orya': 'Odia',
    'pan_Guru': 'Punjabi',
    'san_Deva': 'Sanskrit',
    'sat_Olck': 'Santali',
    'snd_Arab': 'Sindhi (Arabic)',
    'snd_Deva': 'Sindhi (Devanagari)',
    'tam_Taml': 'Tamil',
    'tel_Telu': 'Telugu',
    'urd_Arab': 'Urdu'
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.type !== 'application/pdf') {
        setError('Please select a PDF file');
        return;
      }
      setSelectedFile(file);
      setError(null);
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a PDF file first');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Use the enhanced endpoint with duplicate removal
      const response = await TranslationApiService.translatePdfEnhanced(selectedFile, targetLanguage);
      setResult(response);
      onTranslationComplete?.(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Translation failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadPdf = async () => {
    if (!result || !selectedFile) return;

    try {
      setIsLoading(true);
      await TranslationApiService.downloadTranslatedPdf(
        result.extracted_text,
        result.translated_text,
        result.filename,
        result.target_language
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed');
    } finally {
      setIsLoading(false);
    }
  };

  const resetUpload = () => {
    setSelectedFile(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">PDF Translation</h2>

      {/* File Upload Section */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select PDF File (1-2 pages)
        </label>
        <div className="mb-3 p-3 bg-blue-50 border border-blue-200 rounded-md">
          <p className="text-sm text-blue-800">
            <strong>📄 Supported PDF Types:</strong>
          </p>
          <ul className="text-xs text-blue-700 mt-1 ml-4">
            <li>• Text-based PDFs (fastest processing)</li>
            <li>• Scanned PDFs (OCR will be used)</li>
            <li>• Mixed content PDFs</li>
          </ul>
        </div>
        <div className="flex items-center space-x-4">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            onChange={handleFileSelect}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
          {selectedFile && (
            <button
              onClick={resetUpload}
              className="text-red-600 hover:text-red-800 text-sm"
            >
              Clear
            </button>
          )}
        </div>
        {selectedFile && (
          <p className="text-sm text-gray-600 mt-2">
            Selected: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
          </p>
        )}
      </div>

      {/* Language Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Target Language
        </label>
        <select
          value={targetLanguage}
          onChange={(e) => setTargetLanguage(e.target.value)}
          className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          {Object.entries(languages).map(([code, name]) => (
            <option key={code} value={code}>
              {name}
            </option>
          ))}
        </select>
      </div>

      {/* Upload Button */}
      <button
        onClick={handleUpload}
        disabled={!selectedFile || isLoading}
        className="w-full bg-blue-600 text-white py-3 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
      >
        {isLoading ? 'Translating...' : 'Extract & Translate'}
      </button>

      {/* Error Display */}
      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-md">
          <p className="text-red-700">{error}</p>
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="mt-6 space-y-4">
          <div className="p-4 bg-green-50 border border-green-200 rounded-md">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-green-700 font-medium">
                  ✅ Successfully processed {result.filename} ({result.pages_processed} pages)
                </p>
                {result.extraction_method && (
                  <p className="text-green-600 text-sm mt-1">
                    📋 Extraction method: {result.extraction_method}
                  </p>
                )}
                {result.duplicates_removed && (
                  <p className="text-green-600 text-sm mt-1">
                    🔄 Duplicates removed for cleaner translation
                  </p>
                )}
              </div>
              <button
                onClick={handleDownloadPdf}
                disabled={isLoading}
                className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span>{isLoading ? 'Generating...' : 'Download PDF'}</span>
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h3 className="font-semibold text-gray-800 mb-2">Extracted Text:</h3>
              <div className="p-3 bg-gray-50 border rounded-md max-h-60 overflow-y-auto">
                <p className="text-sm text-gray-700 whitespace-pre-wrap">
                  {result.extracted_text}
                </p>
              </div>
            </div>

            <div>
              <h3 className="font-semibold text-gray-800 mb-2">
                Translated Text ({languages[result.target_language as keyof typeof languages]}):
              </h3>
              <div className="p-3 bg-blue-50 border rounded-md max-h-60 overflow-y-auto">
                <p className="text-sm text-gray-700 whitespace-pre-wrap">
                  {result.translated_text}
                </p>
              </div>
            </div>
          </div>

          {/* Processing Stats */}
          {(result.text_length || result.chunks_processed || result.memory_management) && (
            <div className="p-3 bg-blue-50 border border-blue-200 rounded-md">
              <p className="text-blue-800 text-sm font-medium">Processing Statistics:</p>
              <div className="text-blue-700 text-xs mt-1 space-y-1">
                {result.text_length && <p>• Text length: {result.text_length} characters</p>}
                {result.chunks_processed && <p>• Chunks processed: {result.chunks_processed}</p>}
                {result.memory_management && <p>• Memory optimization: {result.memory_management}</p>}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PdfUpload;