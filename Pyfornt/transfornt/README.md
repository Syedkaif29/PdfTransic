# PDFTransic Frontend

A modern React + TypeScript frontend for translating English text and PDF documents to various Indian languages.

## 🚀 Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Access the app:**
   - Frontend: http://localhost:5173
   - Make sure your backend is running on http://localhost:8000

### Environment Configuration

The app uses environment variables for API configuration:

- **Development**: Uses `http://localhost:8000` (from `.env.local`)
- **Production**: Uses `VITE_API_BASE_URL` environment variable

## 🌐 Deployment

### Vercel Deployment

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Deploy:**
   ```bash
   vercel
   ```

3. **Set environment variable in Vercel dashboard:**
   - `VITE_API_BASE_URL` = `https://syedkaif29-pdftransic.hf.space`

### Netlify Deployment

1. **Install Netlify CLI:**
   ```bash
   npm install -g netlify-cli
   ```

2. **Deploy:**
   ```bash
   netlify deploy --prod
   ```

3. **Set environment variable in Netlify dashboard:**
   - `VITE_API_BASE_URL` = `https://syedkaif29-pdftransic.hf.space`

## 📁 Project Structure

```
src/
├── components/          # React components
│   ├── TranslationForm.tsx
│   ├── PdfUpload.tsx
│   └── EnvChecker.tsx   # Development environment info
├── services/            # API services
│   └── translationApi.ts
├── config/              # Configuration
│   └── api.ts          # API configuration
└── App.tsx             # Main app component
```

## 🔧 Configuration Files

- **`.env.example`** - Template for environment variables
- **`.env.local`** - Local development configuration
- **`.env.production`** - Production configuration template
- **`vercel.json`** - Vercel deployment configuration
- **`netlify.toml`** - Netlify deployment configuration

## 🌍 Features

- **Text Translation**: Direct text input translation
- **PDF Translation**: Upload and translate PDF documents
- **Multi-language Support**: 27+ Indian languages
- **Real-time Health Check**: Backend status monitoring
- **Responsive Design**: Works on desktop and mobile
- **Environment Detection**: Automatic API endpoint switching

## 🔒 Security

- API endpoints are configured via environment variables
- No hardcoded URLs in production builds
- Sensitive configuration excluded from version control

## 🧪 Development Tools

- **Environment Checker**: Shows API configuration in development mode
- **Health Check**: Monitors backend availability
- **Error Handling**: Comprehensive error messages
- **TypeScript**: Full type safety

## 📋 Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## 🔗 API Integration

The frontend connects to the PDFTransic backend API:

- **Health Check**: `GET /health`
- **Languages**: `GET /languages`
- **Text Translation**: `POST /translate-simple`
- **PDF Translation**: `POST /translate-pdf`

## 🚨 Troubleshooting

### API Connection Issues

1. Check environment variables are set correctly
2. Verify backend is running and accessible
3. Check browser console for CORS errors
4. Use the environment checker in development mode

### Build Issues

1. Ensure all dependencies are installed: `npm install`
2. Check TypeScript errors: `npm run lint`
3. Verify environment variables are properly set

## 📄 License

This project is open source and available under the MIT License.