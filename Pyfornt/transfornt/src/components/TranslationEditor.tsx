import React, { useState } from 'react';

interface TranslationEditorProps {
  originalText: string;
  translatedText: string;
  onSave: (editedText: string) => void;
  onCancel: () => void;
  targetLanguage: string;
}

const TranslationEditor: React.FC<TranslationEditorProps> = ({
  originalText,
  translatedText,
  onSave,
  onCancel,
  targetLanguage
}) => {
  const [editedText, setEditedText] = useState(translatedText);
  const [hasChanges, setHasChanges] = useState(false);

  const handleTextChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newText = event.target.value;
    setEditedText(newText);
    setHasChanges(newText !== translatedText);
  };

  const handleSave = () => {
    onSave(editedText);
  };

  const handleReset = () => {
    setEditedText(translatedText);
    setHasChanges(false);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-hidden">
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Edit Translation</h3>
          <p className="text-sm text-gray-600 mt-1">
            Review and edit the translated text before downloading
          </p>
        </div>

        <div className="p-6 overflow-y-auto max-h-[60vh]">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Original Text */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Original Text (English)
              </label>
              <div className="p-3 bg-gray-50 border rounded-md h-64 overflow-y-auto">
                <p className="text-sm text-gray-700 whitespace-pre-wrap">
                  {originalText}
                </p>
              </div>
            </div>

            {/* Translated Text Editor */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Translated Text ({targetLanguage})
                {hasChanges && <span className="text-orange-600 ml-2">• Modified</span>}
              </label>
              <textarea
                value={editedText}
                onChange={handleTextChange}
                className="w-full h-64 p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                placeholder="Edit the translated text here..."
              />
              <div className="mt-2 flex justify-between text-xs text-gray-500">
                <span>Characters: {editedText.length}</span>
                <span>Words: {editedText.split(' ').filter(w => w.trim()).length}</span>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-md">
            <h4 className="text-sm font-medium text-blue-900 mb-2">Quick Actions:</h4>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setEditedText(editedText.charAt(0).toUpperCase() + editedText.slice(1))}
                className="px-3 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
              >
                Capitalize First Letter
              </button>
              <button
                onClick={() => setEditedText(editedText.replace(/\s+/g, ' ').trim())}
                className="px-3 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
              >
                Clean Whitespace
              </button>
              <button
                onClick={() => setEditedText(editedText.replace(/([.!?])\s*([a-z])/g, (_, p1, p2) => p1 + ' ' + p2.toUpperCase()))}
                className="px-3 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
              >
                Fix Sentence Capitalization
              </button>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="p-6 border-t border-gray-200 flex justify-between">
          <div className="flex gap-2">
            <button
              onClick={handleReset}
              disabled={!hasChanges}
              className="px-4 py-2 text-sm text-gray-600 bg-gray-100 rounded-md hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Reset Changes
            </button>
          </div>
          <div className="flex gap-2">
            <button
              onClick={onCancel}
              className="px-4 py-2 text-sm text-gray-600 bg-gray-100 rounded-md hover:bg-gray-200"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-2 text-sm text-white bg-blue-600 rounded-md hover:bg-blue-700"
            >
              Save & Continue
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TranslationEditor;