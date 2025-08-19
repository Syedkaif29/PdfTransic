---
title: PDFTransic Backend
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# PDFTransic Backend

A FastAPI backend for translating English text and PDF documents to various Indian languages using the IndicTrans2 model.

## Features

- **Text Translation**: Direct text input translation to Indian languages
- **PDF Translation**: Extract and translate content from PDF documents  
- **Multi-language Support**: Support for 14+ Indian languages
- **Real-time Processing**: Fast translation using IndicTrans2 model
- **Health Monitoring**: Built-in backend health status monitoring
- **API Documentation**: Interactive API documentation with FastAPI

## API Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check and model status
- `POST /translate` - Translate text to specified Indian language
- `POST /translate-simple` - Simple text translation endpoint
- `POST /translate-pdf` - Extract and translate PDF content
- `GET /languages` - Get list of supported languages
- `GET /docs` - Interactive API documentation

## Supported Languages

The backend supports translation to 27+ Indian languages including:
- Hindi, Tamil, Telugu, Bengali, Gujarati, Marathi, Malayalam, Punjabi, Kannada, Odia, Assamese, Urdu, Sanskrit, Nepali, and more.

## Usage

Once deployed, you can access:
- API Documentation: `https://your-space-url/docs`
- Health Check: `https://your-space-url/health`
- Translation API: `https://your-space-url/translate`

## Model Information

This backend uses the IndicTrans2 model (`ai4bharat/indictrans2-en-indic-dist-200M`) via the IndicTransToolkit for high-quality translations between English and Indian languages.