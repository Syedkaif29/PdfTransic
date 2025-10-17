import React, { useState, useRef, useEffect, useCallback } from 'react';
import { TranslationApiService } from '../services/translationApi';

interface LayoutElement {
  index: number;
  text: string;
  bbox: [number, number, number, number]; // [x, y, width, height]
  font_name: string;
  font_size: number;
  color: [number, number, number];
  is_bold: boolean;
  is_italic: boolean;
  text_align?: string;
  paragraph_index?: number;
  word_index?: number;
  is_title?: boolean;
  is_heading?: boolean;
}

interface LivePreviewEvent {
  type: 'layout_info' | 'translation_update' | 'translation_complete';
  total_elements?: number;
  layout_data?: LayoutElement[];
  element_index?: number;
  original_text?: string;
  translated_text?: string;
  layout?: LayoutElement;
  total_translated?: number;
  progress?: {
    current: number;
    total: number;
    percentage: number;
    batch_progress: string;
  };
}

interface PdfLivePreviewProps {
  file: File;
  targetLanguage: string;
  onClose: () => void;
}

const PdfLivePreview: React.FC<PdfLivePreviewProps> = ({ file, targetLanguage, onClose }) => {
  const [layoutData, setLayoutData] = useState<LayoutElement[]>([]);
  const [translatedElements, setTranslatedElements] = useState<Map<number, string>>(new Map());
  const [isStreaming, setIsStreaming] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isComplete, setIsComplete] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const eventSourceRef = useRef<XMLHttpRequest | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const startLivePreview = useCallback(async () => {
    try {
      setIsStreaming(true);
      setError(null);
      setProgress(0);
      setTranslatedElements(new Map());

      // Create a mock PDF canvas for visualization
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          // Set canvas size
          canvas.width = 600;
          canvas.height = 800;
          
          // Clear canvas
          ctx.fillStyle = '#ffffff';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          
          // Draw PDF-like border
          ctx.strokeStyle = '#cccccc';
          ctx.lineWidth = 2;
          ctx.strokeRect(10, 10, canvas.width - 20, canvas.height - 20);
          
          // Add title
          ctx.fillStyle = '#333333';
          ctx.font = 'bold 16px Arial';
          ctx.fillText('PDF Live Translation Preview', 20, 40);
        }
      }

      // Start streaming translation
      const xhr = TranslationApiService.streamLivePreview(
        file,
        targetLanguage,
        (event: LivePreviewEvent) => {
          if (event.type === 'layout_info') {
            setLayoutData(event.layout_data || []);
            drawLayoutElements(event.layout_data || []);
          } else if (event.type === 'translation_update') {
            if (event.element_index !== undefined && event.translated_text) {
              setTranslatedElements(prev => {
                const newMap = new Map(prev);
                newMap.set(event.element_index!, event.translated_text!);
                return newMap;
              });
              
              // Update progress with detailed information
              if (event.progress) {
                setProgress(event.progress.percentage);
              } else {
                const totalElements = layoutData.length || 1;
                setProgress(((event.element_index! + 1) / totalElements) * 100);
              }
              
              // Update canvas with translated text
              updateCanvasWithTranslation(event.element_index!, event.translated_text!, event.layout);
            }
          } else if (event.type === 'translation_complete') {
            setIsStreaming(false);
            setProgress(100);
            setIsComplete(true);
          } else {
            // Handle legacy format for backward compatibility
            if (event.element_index !== undefined && event.translated_text) {
              setTranslatedElements(prev => {
                const newMap = new Map(prev);
                newMap.set(event.element_index!, event.translated_text!);
                return newMap;
              });
              
              const totalElements = layoutData.length || 1;
              setProgress(((event.element_index! + 1) / totalElements) * 100);
              
              // Update canvas with translated text (no layout info in legacy format)
              updateCanvasWithTranslation(event.element_index!, event.translated_text!, null);
            }
          }
        },
        (error: Error) => {
          setError(error.message);
          setIsStreaming(false);
        },
        () => {
          setIsStreaming(false);
        }
      );
      
      eventSourceRef.current = xhr;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      setIsStreaming(false);
    }
  }, [file, targetLanguage, layoutData.length]);

  useEffect(() => {
    startLivePreview();
    return () => {
      const eventSource = eventSourceRef.current;
      if (eventSource) {
        eventSource.abort();
      }
    };
  }, [startLivePreview]);

  const drawLayoutElements = (elements: LayoutElement[]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Calculate canvas size based on content
    const maxY = Math.max(...elements.map(el => el.bbox[3]), 600);
    const canvasHeight = Math.max(maxY + 100, 800);
    canvas.height = canvasHeight;

    // Clear and redraw background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw document-like border
    ctx.strokeStyle = '#333333';
    ctx.lineWidth = 1;
    ctx.strokeRect(20, 20, canvas.width - 40, canvas.height - 40);
    
    // Add document header
    ctx.fillStyle = '#333333';
    ctx.font = 'bold 18px Arial';
    ctx.fillText('Document Translation Preview', 30, 50);
    
    // Add subtitle
    ctx.fillStyle = '#666666';
    ctx.font = '12px Arial';
    ctx.fillText('Original text in gray, translated text in green', 30, 70);

    // Group elements by paragraph for better rendering
    const paragraphGroups = new Map<number, LayoutElement[]>();
    elements.forEach(element => {
      const paraIndex = element.paragraph_index || 0;
      if (!paragraphGroups.has(paraIndex)) {
        paragraphGroups.set(paraIndex, []);
      }
      paragraphGroups.get(paraIndex)!.push(element);
    });

    // Draw each paragraph with proper formatting
    paragraphGroups.forEach((paragraphElements) => {
      if (paragraphElements.length === 0) return;
      
      const firstElement = paragraphElements[0];
      const isTitle = firstElement.is_title;
      const isHeading = firstElement.is_heading;
      
      // Draw paragraph background (subtle)
      if (isTitle || isHeading) {
        ctx.fillStyle = 'rgba(240, 240, 240, 0.3)';
        const minX = Math.min(...paragraphElements.map(el => el.bbox[0]));
        const minY = Math.min(...paragraphElements.map(el => el.bbox[1]));
        const maxX = Math.max(...paragraphElements.map(el => el.bbox[2]));
        const maxY = Math.max(...paragraphElements.map(el => el.bbox[3]));
        ctx.fillRect(minX - 5, minY - 5, maxX - minX + 10, maxY - minY + 10);
      }
      
      // Draw each word in the paragraph
      paragraphElements.forEach((element) => {
        const [x, y, width, height] = element.bbox;
        
        // Draw original text with proper styling
        ctx.fillStyle = '#666666';
        ctx.font = `${element.is_bold ? 'bold' : 'normal'} ${element.font_size}px ${element.font_name}`;
        
        // Apply text alignment
        let textX = x + 2;
        if (element.text_align === 'center') {
          textX = canvas.width / 2 - (ctx.measureText(element.text).width / 2);
        } else if (element.text_align === 'right') {
          textX = x + width - ctx.measureText(element.text).width - 2;
        }
        
        ctx.fillText(element.text, textX, y + element.font_size + 2);
        
        // Add subtle border for individual words (lighter for better readability)
        ctx.strokeStyle = '#f5f5f5';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x, y, width, height);
      });
    });
  };

  const updateCanvasWithTranslation = (index: number, translatedText: string, layout?: LayoutElement | null) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (layout) {
      const [x, y, width, height] = layout.bbox;
      
      // Clear the area
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(x, y, width, height);
      
      // Draw translated text with proper alignment
      ctx.fillStyle = '#2E7D32';
      ctx.font = `${layout.is_bold ? 'bold' : 'normal'} ${layout.font_size}px ${layout.font_name}`;
      
      // Apply text alignment for translated text
      let textX = x + 2;
      if (layout.text_align === 'center') {
        textX = canvas.width / 2 - (ctx.measureText(translatedText).width / 2);
      } else if (layout.text_align === 'right') {
        textX = x + width - ctx.measureText(translatedText).width - 2;
      }
      
      ctx.fillText(translatedText, textX, y + layout.font_size + 2);
      
      // Add green border (subtle)
      ctx.strokeStyle = '#4CAF50';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(x, y, width, height);
      
      // Add subtle highlight effect
      ctx.fillStyle = 'rgba(76, 175, 80, 0.03)';
      ctx.fillRect(x, y, width, height);
      
      // Add completion indicator (smaller and less intrusive)
      ctx.fillStyle = '#4CAF50';
      ctx.font = '6px Arial';
      ctx.fillText('✓', x + width - 8, y + 8);
    } else {
      // Fallback for when no layout is available - just add a simple indicator
      const y = 100 + (index * 20); // Simple vertical stacking
      ctx.fillStyle = '#2E7D32';
      ctx.font = '12px Arial';
      ctx.fillText(`${index}: ${translatedText.substring(0, 50)}...`, 30, y);
      
      // Add completion indicator
      ctx.fillStyle = '#4CAF50';
      ctx.font = '10px Arial';
      ctx.fillText('✓', 10, y);
    }
  };

  const handleDownloadPDF = async () => {
    if (!file) return;
    
    try {
      setIsDownloading(true);
      
      // Prepare data for PDF download
      const originalTexts = Array.from(translatedElements.entries()).map(([index]) => {
        const element = layoutData[index];
        return element ? element.text : '';
      });
      
      const translatedTexts = Array.from(translatedElements.values());
      
      // Call the download PDF endpoint
      const formData = new FormData();
      formData.append('file', file);
      formData.append('target_language', targetLanguage);
      formData.append('original_text_chunks_json', JSON.stringify(originalTexts));
      formData.append('translated_text_chunks_json', JSON.stringify(translatedTexts));
      formData.append('layout_data_json', JSON.stringify(layoutData));
      formData.append('language_code', targetLanguage);
      
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/download-translated-pdf`, {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `translated_${file.name}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        throw new Error('Failed to download PDF');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed');
    } finally {
      setIsDownloading(false);
    }
  };

  const handleDownloadWord = async () => {
    if (!file) return;
    
    try {
      setIsDownloading(true);
      
      // Prepare data for Word download
      const originalTexts = Array.from(translatedElements.entries()).map(([index]) => {
        const element = layoutData[index];
        return element ? element.text : '';
      });
      
      const translatedTexts = Array.from(translatedElements.values());
      
      // Call the download Word endpoint
      const formData = new FormData();
      formData.append('file', file);
      formData.append('target_language', targetLanguage);
      formData.append('original_text_chunks_json', JSON.stringify(originalTexts));
      formData.append('translated_text_chunks_json', JSON.stringify(translatedTexts));
      formData.append('layout_data_json', JSON.stringify(layoutData));
      formData.append('language_code', targetLanguage);
      
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/download-translated-word`, {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `translated_${file.name.replace('.pdf', '.docx')}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        throw new Error('Failed to download Word document');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed');
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-7xl w-full max-h-[95vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex justify-between items-center p-4 border-b">
          <h2 className="text-xl font-bold text-gray-800">
            Live PDF Translation Preview
          </h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 text-2xl"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 p-6 overflow-auto">
          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
              {error}
            </div>
          )}

          {/* Progress Bar */}
          <div className="mb-6">
            <div className="flex justify-between items-center mb-3">
              <span className="text-lg font-semibold text-gray-800">
                Translation Progress
              </span>
              <span className="text-lg font-bold text-blue-600">
                {Math.round(progress)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            {layoutData.length > 0 && (
              <div className="text-sm text-gray-600 mt-2">
                Processing {layoutData.length} text elements from document
              </div>
            )}
          </div>

          {/* PDF Preview Canvas */}
          <div className="flex justify-center mb-6">
            <div className="border-2 border-gray-300 rounded-lg overflow-auto shadow-xl max-h-[500px] bg-white">
              <canvas
                ref={canvasRef}
                width={700}
                height={900}
                className="block"
              />
            </div>
          </div>

          {/* Translation Status */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <div className="bg-gray-50 p-5 rounded-lg border">
              <h3 className="font-semibold text-gray-800 mb-3 text-lg">Original Text</h3>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {layoutData.map((element, index) => (
                  <div key={index} className="text-sm text-gray-600 flex items-start">
                    <span className="font-mono text-xs bg-gray-200 px-2 py-1 rounded mr-2 flex-shrink-0">
                      {index}
                    </span>
                    <span className="break-words">{element.text}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-green-50 p-5 rounded-lg border">
              <h3 className="font-semibold text-gray-800 mb-3 text-lg">Translated Text</h3>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {Array.from(translatedElements.entries()).map(([index, text]) => (
                  <div key={index} className="text-sm text-green-700 flex items-start">
                    <span className="font-mono text-xs bg-green-200 px-2 py-1 rounded mr-2 flex-shrink-0">
                      {index}
                    </span>
                    <span className="break-words">{text}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Download Section */}
          {isComplete && (
            <div className="bg-gradient-to-r from-green-50 to-blue-50 p-6 rounded-lg border-2 border-green-200 mb-6">
              <div className="text-center">
                <div className="flex items-center justify-center mb-4">
                  <div className="bg-green-100 p-3 rounded-full mr-3">
                    <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <h3 className="text-2xl font-bold text-gray-800">Translation Complete!</h3>
                </div>
                <p className="text-gray-600 mb-6">Your document has been successfully translated. Download it in your preferred format:</p>
                
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <button
                    onClick={handleDownloadPDF}
                    disabled={isDownloading}
                    className="flex items-center justify-center px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-gray-400 transition-colors font-semibold"
                  >
                    {isDownloading ? (
                      <>
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                        Downloading...
                      </>
                    ) : (
                      <>
                        <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        Download PDF
                      </>
                    )}
                  </button>
                  
                  <button
                    onClick={handleDownloadWord}
                    disabled={isDownloading}
                    className="flex items-center justify-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition-colors font-semibold"
                  >
                    {isDownloading ? (
                      <>
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                        Downloading...
                      </>
                    ) : (
                      <>
                        <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        Download Word
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Status */}
          {!isComplete && (
            <div className="text-center">
              {isStreaming && (
                <div className="flex items-center justify-center space-x-3">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                  <span className="text-lg text-gray-600 font-medium">Translating document...</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PdfLivePreview;
