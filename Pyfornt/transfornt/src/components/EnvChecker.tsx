import React from 'react';
import { ENV_INFO } from '../config/api';

const EnvChecker: React.FC = () => {
  return (
    <div className="fixed bottom-4 right-4 bg-gray-800 text-white p-3 rounded-lg text-xs max-w-xs">
      <div className="font-bold mb-2">🔧 API Info</div>
      <div>Mode: {ENV_INFO.mode}</div>
      <div>API: {ENV_INFO.apiBaseUrl}</div>
      <div>Env: {ENV_INFO.isDevelopment ? 'Development' : 'Production'}</div>
      <div>Var: {import.meta.env.VITE_API_BASE_URL || 'Not set'}</div>
    </div>
  );
};

export default EnvChecker;