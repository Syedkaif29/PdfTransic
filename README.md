# PDFTransic

A full-stack application for translating English text and PDF documents to various Indian languages using the IndicTrans2 model.

## 🎯 Project Overview

PDFTransic is designed to bridge language barriers by providing seamless translation of both text and PDF documents into multiple Indian languages. The application leverages the powerful IndicTrans2 model for accurate translations while offering a modern, user-friendly interface.

## 📸 Screenshots

### Text Translation Interface
![Text Translation](Pyfornt/transfornt/public/pdftrans1.png)

### PDF Translation Interface  
![PDF Translation](Pyfornt/transfornt/public/trans2.png)

## ✨ Key Features

- **📝 Text Translation**: Direct text input translation to Indian languages
- **📄 PDF Translation**: Extract and translate content from PDF documents
- **🌐 Multi-language Support**: Support for 14+ Indian languages including Hindi, Tamil, Telugu, Bengali, and more
- **⚡ Real-time Processing**: Fast translation using IndicTrans2 model
- **🎨 Modern UI**: Clean and responsive React interface with Tailwind CSS
- **📊 Health Monitoring**: Built-in backend health status monitoring
- **📚 API Documentation**: Interactive API documentation with FastAPI

## 🏗️ Architecture

- **Backend**: FastAPI with IndicTrans2 model integration (Pyback1/)
- **Frontend**: React with TypeScript and Tailwind CSS (Pyfornt/transfornt/)
- **Translation Engine**: IndicTrans2 via IndicTransToolkit
- **PDF Processing**: Built-in PDF text extraction capabilities

## 🚀 Setup Instructions

### Backend Setup (FastAPI)

1. Navigate to the backend directory:

```bash
cd Pyback1
```

2. Create and activate virtual environment:

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the server:

```bash
python start_server.py
```

The server will be available at `http://localhost:8000` with API documentation at `http://localhost:8000/docs`.

### Frontend Setup (React + TypeScript)

1. Navigate to the frontend directory:

```bash
cd Pyfornt/transfornt
```

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`.

## 🌍 Supported Languages

PDFTransic supports translation to the following Indian languages:

- **Hindi** (hin_Deva)
- **Tamil** (tam_Taml)
- **Telugu** (tel_Telu)
- **Bengali** (ben_Beng)
- **Gujarati** (guj_Gujr)
- **Marathi** (mar_Deva)
- **Malayalam** (mal_Mlym)
- **Punjabi** (pan_Guru)
- **Kannada** (kan_Knda)
- **Odia** (ory_Orya)
- **Assamese** (asm_Beng)
- **Urdu** (urd_Arab)
- **Sanskrit** (san_Deva)
- **Nepali** (npi_Deva)

## 🔧 API Endpoints

- `GET /health` - Health check and model status
- `POST /translate` - Translate text to specified Indian language
- `POST /translate-pdf` - Extract and translate PDF content
- `GET /docs` - Interactive API documentation

## 💻 Development

### Backend Development

- Main FastAPI application: `Pyback1/main.py`
- Server startup script: `Pyback1/start_server.py`
- Dependencies: `Pyback1/requirements.txt`

### Frontend Development

- React components: `Pyfornt/transfornt/src/components/`
- Translation API service: `Pyfornt/transfornt/src/services/translationApi.ts`
- Main app component: `Pyfornt/transfornt/src/App.tsx`

## 📋 System Requirements

- **Python**: 3.8 or higher
- **Node.js**: 16 or higher
- **RAM**: At least 4GB (for model loading)
- **Storage**: 2GB+ free space (for models)
- **OS**: Windows, macOS, or Linux

## ⚠️ Important Notes

- **First Startup**: Model loading may take 3-5 minutes on first startup
- **Health Check**: Monitor the `/health` endpoint to verify when models are ready
- **PDF Limitations**: Optimized for 1-2 page PDFs for best performance
- **Model Dependencies**: Uses IndicTransToolkit for translation capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.