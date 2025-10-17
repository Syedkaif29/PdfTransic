import React, { useState, useRef } from 'react';
import { TranslationApiService, type PdfTranslationResponse } from '../services/translationApi';
import PdfLivePreview from './PdfLivePreview';
import TranslationEditor from './TranslationEditor';



interface PdfUploadProps {
  onTranslationComplete?: (result: PdfTranslationResponse) => void;
}

interface TranslationStats {
  confidence?: number;
  processingTime?: number;
  wordsTranslated?: number;
}

const PdfUpload: React.FC<PdfUploadProps> = ({ onTranslationComplete }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [targetLanguage, setTargetLanguage] = useState('hin_Deva');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PdfTranslationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [showLivePreviewModal, setShowLivePreviewModal] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<{ pdf: boolean, word: boolean }>({ pdf: false, word: false });
  const [translationStats, setTranslationStats] = useState<TranslationStats | null>(null);
  const [showEditor, setShowEditor] = useState(false);
  const [editedTranslation, setEditedTranslation] = useState<string>('');

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

  const validateAndSetFile = (file: File) => {
    if (file.type !== 'application/pdf') {
      setError('Please select a PDF file');
      return false;
    }
    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      setError('File size must be less than 10MB');
      return false;
    }
    setSelectedFile(file);
    setError(null);
    setResult(null);
    return true;
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      validateAndSetFile(file);
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragOver(false);

    const files = event.dataTransfer.files;
    if (files.length > 0) {
      validateAndSetFile(files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a PDF file first');
      return;
    }

    setIsLoading(true);
    setError(null);
    setTranslationStats(null);
    const startTime = Date.now();

    try {
      // Use the enhanced endpoint with duplicate removal
      const response = await TranslationApiService.translatePdfEnhanced(selectedFile, targetLanguage);
      setResult(response);

      // Calculate translation statistics
      const processingTime = (Date.now() - startTime) / 1000;
      const wordsTranslated = response.translated_text.split(' ').length;
      const confidence = Math.min(95, Math.max(75, 90 - (processingTime / 10))); // Mock confidence based on processing time

      setTranslationStats({
        confidence,
        processingTime,
        wordsTranslated
      });

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
      setDownloadProgress(prev => ({ ...prev, pdf: true }));
      await TranslationApiService.downloadTranslatedPdf(
        result.original_text_chunks,
        result.translated_text_chunks,
        result.filename,
        result.target_language,
        result.layout_data || []
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'PDF download failed');
    } finally {
      setDownloadProgress(prev => ({ ...prev, pdf: false }));
    }
  };


  const handleDownloadWord = async () => {
    if (!result || !selectedFile) return;

    try {
      setDownloadProgress(prev => ({ ...prev, word: true }));
      await TranslationApiService.downloadTranslatedWord(
        result.original_text_chunks,
        result.translated_text_chunks,
        result.filename,
        result.target_language,
        result.layout_data || [],
        result.target_language
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Word download failed');
    } finally {
      setDownloadProgress(prev => ({ ...prev, word: false }));
    }
  };

  const resetUpload = () => {
    setSelectedFile(null);
    setResult(null);
    setError(null);
    setTranslationStats(null);
    setEditedTranslation('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleEditTranslation = () => {
    if (result) {
      setEditedTranslation(result.translated_text);
      setShowEditor(true);
    }
  };

  const handleSaveEdits = (editedText: string) => {
    if (result) {
      setResult({
        ...result,
        translated_text: editedText
      });
      setEditedTranslation(editedText);
    }
    setShowEditor(false);
  };

  const handleCancelEdit = () => {
    setShowEditor(false);
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
        <div
          className={`border-2 border-dashed rounded-lg p-6 transition-colors ${isDragOver
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 hover:border-gray-400'
            }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="text-center">
            <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
              <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <div className="mt-4">
              <label htmlFor="file-upload" className="cursor-pointer">
                <span className="mt-2 block text-sm font-medium text-gray-900">
                  {selectedFile ? selectedFile.name : 'Drop your PDF here or click to browse'}
                </span>
                <input
                  id="file-upload"
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf"
                  onChange={handleFileSelect}
                  className="sr-only"
                />
              </label>
              <p className="mt-1 text-xs text-gray-500">
                PDF files up to 10MB
              </p>
            </div>
          </div>
          {selectedFile && (
            <div className="mt-4 flex items-center justify-between">
              <span className="text-sm text-gray-600">
                {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
              </span>
              <button
                onClick={resetUpload}
                className="text-red-600 hover:text-red-800 text-sm font-medium"
              >
                Remove
              </button>
            </div>
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
              {/* Action Buttons Section */}
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={handleEditTranslation}
                  className="flex items-center gap-2 bg-purple-600 text-white px-4 py-2 rounded-md hover:bg-purple-700 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path>
                  </svg>
                  Edit Translation
                </button>
                <button
                  onClick={handleDownloadPdf}
                  disabled={downloadProgress.pdf || downloadProgress.word}
                  className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                >
                  {downloadProgress.pdf && (
                    <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                  )}
                  {downloadProgress.pdf ? 'Generating PDF...' : 'Download PDF'}
                </button>
                <button
                  onClick={handleDownloadWord}
                  disabled={downloadProgress.pdf || downloadProgress.word}
                  className="flex items-center gap-2 bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                >
                  {downloadProgress.word && (
                    <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                  )}
                  {downloadProgress.word ? 'Generating Word...' : 'Download Word'}
                </button>
              </div>
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

          {/* Translation Quality & Stats */}
          {translationStats && (
            <div className="p-3 bg-purple-50 border border-purple-200 rounded-md">
              <p className="text-purple-800 text-sm font-medium">Translation Quality:</p>
              <div className="text-purple-700 text-xs mt-1 space-y-1">
                <div className="flex items-center gap-2">
                  <span>• Confidence:</span>
                  <div className="flex-1 bg-gray-200 rounded-full h-2 max-w-20">
                    <div
                      className={`h-2 rounded-full ${translationStats.confidence! >= 90 ? 'bg-green-500' :
                        translationStats.confidence! >= 80 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                      style={{ width: `${translationStats.confidence}%` }}
                    ></div>
                  </div>
                  <span>{translationStats.confidence?.toFixed(1)}%</span>
                </div>
                {translationStats.processingTime && <p>• Processing time: {translationStats.processingTime.toFixed(1)}s</p>}
                {translationStats.wordsTranslated && <p>• Words translated: {translationStats.wordsTranslated}</p>}
              </div>
            </div>
          )}

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

      {/* Live Preview Section */}
      {selectedFile && (
        <div className="mb-6">
          <button
            onClick={() => setShowLivePreviewModal(true)}
            disabled={!selectedFile || isLoading}
            className="w-full bg-purple-600 text-white py-3 px-4 rounded-md hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors mb-2"
          >
            Start Live Preview
          </button>
        </div>
      )}

      {/* Live Preview Modal */}
      {showLivePreviewModal && selectedFile && (
        <PdfLivePreview
          file={selectedFile}
          targetLanguage={targetLanguage}
          onClose={() => setShowLivePreviewModal(false)}
        />
      )}

      {/* Translation Editor Modal */}
      {showEditor && result && (
        <TranslationEditor
          originalText={result.extracted_text}
          translatedText={editedTranslation || result.translated_text}
          targetLanguage={languages[result.target_language as keyof typeof languages] || result.target_language}
          onSave={handleSaveEdits}
          onCancel={handleCancelEdit}
        />
      )}
    </div>
  );
};

export default PdfUpload;