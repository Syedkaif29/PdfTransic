#!/usr/bin/env python3
"""
Deploy the memory-optimized PDF translation service
"""
import os
import shutil
import subprocess
import sys

def deploy_memory_fix():
    """Deploy the updated service with memory optimizations"""
    
    print("🚀 Deploying Memory-Optimized PDF Translation Service...")
    print("=" * 60)
    
    # Create deployment directory
    deploy_dir = "hf_space_memory_fix"
    
    if os.path.exists(deploy_dir):
        print(f"📁 Removing existing {deploy_dir}...")
        shutil.rmtree(deploy_dir)
    
    print(f"📁 Creating {deploy_dir}...")
    os.makedirs(deploy_dir)
    
    # Copy main application
    print("📄 Copying main.py...")
    shutil.copy2("Pyback1/main.py", f"{deploy_dir}/main.py")
    
    # Copy requirements
    print("📄 Copying requirements.txt...")
    shutil.copy2("Pyback1/requirements.txt", f"{deploy_dir}/requirements.txt")
    
    # Create optimized Dockerfile
    dockerfile_content = '''FROM python:3.11-slim

# Install system dependencies including Tesseract OCR
RUN apt-get update && apt-get install -y \\
    tesseract-ocr \\
    tesseract-ocr-eng \\
    libtesseract-dev \\
    poppler-utils \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY main.py .

# Set environment variables for memory optimization
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/huggingface_cache

# Create cache directories
RUN mkdir -p /tmp/transformers_cache /tmp/huggingface_cache

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:7860/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
'''
    
    print("📄 Creating optimized Dockerfile...")
    with open(f"{deploy_dir}/Dockerfile", "w", encoding="utf-8") as f:
        f.write(dockerfile_content)
    
    # Create startup script
    startup_content = '''#!/bin/bash
set -e

echo "🚀 Starting Memory-Optimized PDF Translation Service..."

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_HOME=/tmp/huggingface_cache

# Create cache directories
mkdir -p /tmp/transformers_cache /tmp/huggingface_cache

# Start the application
uvicorn main:app --host 0.0.0.0 --port 7860 --workers 1
'''
    
    print("📄 Creating startup.sh...")
    with open(f"{deploy_dir}/startup.sh", "w", encoding="utf-8") as f:
        f.write(startup_content)
    
    # Make startup script executable
    os.chmod(f"{deploy_dir}/startup.sh", 0o755)
    
    # Create README
    readme_content = '''# Memory-Optimized PDF Translation Service

This is an enhanced version of the IndicTrans2 PDF translation service with memory optimizations to handle larger PDFs and prevent out-of-memory errors.

## Key Improvements

### Memory Management
- **Text Chunking**: Large PDFs are split into smaller chunks for processing
- **Batch Processing**: Chunks are processed in small batches to manage memory
- **GPU Memory Clearing**: Memory is cleared between batches
- **Optimized Model Loading**: Uses float16 precision and low memory usage flags
- **Reduced Beam Search**: Uses 3 beams instead of 5 to save memory

### PDF Processing
- **Enhanced OCR**: Improved OCR with image resizing for memory efficiency
- **Multiple Extraction Methods**: PyPDF2, PyMuPDF, and OCR fallback
- **Memory-Aware Image Processing**: Automatic image resizing for large PDFs

### New Endpoints
- `POST /clear-memory`: Clear GPU memory cache
- `GET /memory-info`: Get current memory usage information

## Usage

### Simple Text Translation
```bash
curl -X POST "https://your-space.hf.space/translate-simple" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "Hello world", "target_language": "hin_Deva"}'
```

### PDF Translation
```bash
curl -X POST "https://your-space.hf.space/translate-pdf" \\
     -F "file=@document.pdf" \\
     -F "target_language=hin_Deva"
```

### Memory Management
```bash
# Check memory usage
curl "https://your-space.hf.space/memory-info"

# Clear memory cache
curl -X POST "https://your-space.hf.space/clear-memory"
```

## Supported Languages
- Hindi (hin_Deva)
- Bengali (ben_Beng)
- Tamil (tam_Taml)
- Telugu (tel_Telu)
- And 20+ other Indian languages

## Error Handling
The service now handles memory errors gracefully by:
1. Automatically chunking large texts
2. Processing in smaller batches
3. Clearing memory between operations
4. Providing detailed error messages
'''
    
    print("📄 Creating README.md...")
    with open(f"{deploy_dir}/README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # Create .gitignore
    gitignore_content = '''__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis
'''
    
    print("📄 Creating .gitignore...")
    with open(f"{deploy_dir}/.gitignore", "w", encoding="utf-8") as f:
        f.write(gitignore_content)
    
    print("\n✅ Deployment package created successfully!")
    print(f"📁 Location: {deploy_dir}/")
    print("\n🚀 Next steps:")
    print("1. Test locally: cd {deploy_dir} && python main.py")
    print("2. Deploy to Hugging Face Spaces")
    print("3. Monitor memory usage with /memory-info endpoint")
    print("\n💡 Memory Optimizations Applied:")
    print("✅ Text chunking for large PDFs")
    print("✅ Batch processing (3 chunks at a time)")
    print("✅ GPU memory clearing between batches")
    print("✅ Float16 precision for GPU models")
    print("✅ Reduced beam search (3 instead of 5)")
    print("✅ OCR image resizing")
    print("✅ Memory monitoring endpoints")

if __name__ == "__main__":
    deploy_memory_fix()