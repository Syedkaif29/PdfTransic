# Indian Language Translator

A full-stack application for translating English text to various Indian languages using IndicTrans2 model.

## Architecture

- **Backend**: FastAPI with IndicTrans2 model (Pyback1/)
- **Frontend**: React with TypeScript (Pyfornt/transfornt/)

## Setup Instructions

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

## Features

- **Multi-language Support**: Translate English text to various Indian languages
- **Real-time Translation**: Fast translation using IndicTrans2 model
- **Modern UI**: Clean and responsive React interface with Tailwind CSS
- **API Documentation**: Interactive API docs with FastAPI
- **Health Monitoring**: Built-in health check endpoints

## API Endpoints

- `GET /health` - Health check and model status
- `POST /translate` - Translate text to specified Indian language
- `GET /docs` - Interactive API documentation

## Supported Languages

The application supports translation to various Indian languages through the IndicTrans2 model. Check the API documentation for the complete list of supported language codes.

## Development

### Backend Development

- The main FastAPI application is in `Pyback1/main.py`
- Server startup script: `Pyback1/start_server.py`
- Dependencies: `Pyback1/requirements.txt`

### Frontend Development

- React components in `Pyfornt/transfornt/src/components/`
- Translation API service: `Pyfornt/transfornt/src/services/translationApi.ts`
- Main app component: `Pyfornt/transfornt/src/App.tsx`

## Requirements

- Python 3.8+
- Node.js 16+
- PyTorch (for IndicTrans2 model)
- At least 4GB RAM (model loading)

## Notes

- Model loading may take a few minutes on first startup
- Check the `/health` endpoint to verify when models are ready
- The application uses IndicTransToolkit for translation capabilities
