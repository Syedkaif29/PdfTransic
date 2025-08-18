#!/usr/bin/env python3
"""
Simple script to start the FastAPI server
"""
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting IndicTrans2 Translation API Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔄 Health Check: http://localhost:8000/health")
    print("\n⚠️  Note: Model loading may take a few minutes on first startup")
    print("💡 Check /health endpoint to see when models are ready\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )