import React from 'react';
import { ENV_INFO } from '../config/api';

const EnvChecker: React.FC = () => {
  // Only show in development mode
  if (!ENV_INFO.isDevelopment) {
    return null;
  }

  return (
    <div className="fixed bottom-4 right-4 bg-gray-800 text-white p-3 rounded-lg text-xs max-w-xs">
      <div className="font-bold mb-2">🔧 Dev Info</div>
      <div>Mode: {ENV_INFO.mode}</div>
      <div>API: {ENV_INFO.apiBaseUrl}</div>
      <div>Env: {ENV_INFO.isDevelopment ? 'Development' : 'Production'}</div>
    </div>
  );
};

export default EnvChecker;