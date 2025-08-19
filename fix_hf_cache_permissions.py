#!/usr/bin/env python3
"""
Fix cache permission issues for HF Spaces deployment
"""
import os
import shutil

def fix_cache_permissions():
    """Fix cache permission issues in all HF spaces"""
    
    print("🔧 Fixing cache permission issues for HF Spaces...")
    
    # Updated Dockerfile with proper cache handling
    dockerfile_content = '''FROM python:3.11-slim

# Set environment variables for cache directories
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/huggingface_cache
ENV TORCH_HOME=/tmp/torch_cache
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    tesseract-ocr \\
    tesseract-ocr-eng \\
    libtesseract-dev \\
    poppler-utils \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create cache directories with proper permissions
RUN mkdir -p /tmp/transformers_cache /tmp/huggingface_cache /tmp/torch_cache && \\
    chmod -R 777 /tmp/transformers_cache /tmp/huggingface_cache /tmp/torch_cache

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set proper permissions for the app directory
RUN chmod -R 755 /app

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && \\
    chown -R appuser:appuser /app /tmp/transformers_cache /tmp/huggingface_cache /tmp/torch_cache

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \\
    CMD curl -f http://localhost:7860/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
'''

    # Cache-aware main.py additions
    cache_fix_code = '''
import os
import tempfile

# Set up cache directories with fallback
def setup_cache_dirs():
    """Setup cache directories with proper permissions"""
    cache_dirs = [
        os.environ.get('TRANSFORMERS_CACHE', '/tmp/transformers_cache'),
        os.environ.get('HF_HOME', '/tmp/huggingface_cache'),
        os.environ.get('TORCH_HOME', '/tmp/torch_cache')
    ]
    
    for cache_dir in cache_dirs:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(cache_dir, 'test_write')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info(f"✅ Cache directory ready: {cache_dir}")
        except Exception as e:
            logger.warning(f"⚠️ Cache directory issue {cache_dir}: {e}")
            # Fallback to temp directory
            fallback_dir = tempfile.mkdtemp()
            if 'TRANSFORMERS_CACHE' in cache_dir:
                os.environ['TRANSFORMERS_CACHE'] = fallback_dir
            elif 'HF_HOME' in cache_dir:
                os.environ['HF_HOME'] = fallback_dir
            elif 'TORCH_HOME' in cache_dir:
                os.environ['TORCH_HOME'] = fallback_dir
            logger.info(f"📁 Using fallback cache: {fallback_dir}")

# Call setup at startup
setup_cache_dirs()
'''

    # Update all HF spaces
    hf_spaces = [
        "hf_space_deploy",
        "hf_space_fix", 
        "hf_space_memory_fix",
        "hf_space_requirements_fix"
    ]
    
    for space in hf_spaces:
        print(f"\n🔧 Fixing {space}...")
        
        # Update Dockerfile
        dockerfile_path = f"{space}/Dockerfile"
        try:
            with open(dockerfile_path, 'w', encoding='utf-8') as f:
                f.write(dockerfile_content)
            print(f"✅ Updated {space}/Dockerfile")
        except Exception as e:
            print(f"❌ Failed to update Dockerfile for {space}: {e}")
        
        # Update main.py with cache fix
        main_py_path = f"{space}/main.py"
        try:
            # Read existing main.py
            with open(main_py_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add cache setup after imports if not already present
            if 'setup_cache_dirs' not in content:
                # Find the position after imports
                lines = content.split('\n')
                insert_pos = 0
                
                for i, line in enumerate(lines):
                    if line.startswith('from ') or line.startswith('import '):
                        insert_pos = i + 1
                    elif line.strip() == '' and insert_pos > 0:
                        continue
                    elif insert_pos > 0 and not (line.startswith('from ') or line.startswith('import ') or line.strip() == ''):
                        break
                
                # Insert cache setup code
                lines.insert(insert_pos, cache_fix_code)
                
                # Write back
                with open(main_py_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                
                print(f"✅ Added cache fix to {space}/main.py")
            else:
                print(f"ℹ️  Cache fix already present in {space}/main.py")
                
        except Exception as e:
            print(f"❌ Failed to update main.py for {space}: {e}")

def create_startup_script():
    """Create startup script for HF Spaces"""
    
    startup_content = '''#!/bin/bash
set -e

echo "🚀 Starting IndicTrans2 Translation Service..."

# Set cache environment variables
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_HOME=/tmp/huggingface_cache
export TORCH_HOME=/tmp/torch_cache

# Create cache directories
mkdir -p $TRANSFORMERS_CACHE $HF_HOME $TORCH_HOME
chmod -R 777 $TRANSFORMERS_CACHE $HF_HOME $TORCH_HOME

echo "📁 Cache directories ready"
echo "🔧 Starting FastAPI server..."

# Start the application
exec uvicorn main:app --host 0.0.0.0 --port 7860 --workers 1
'''
    
    hf_spaces = [
        "hf_space_deploy",
        "hf_space_fix", 
        "hf_space_memory_fix",
        "hf_space_requirements_fix"
    ]
    
    for space in hf_spaces:
        startup_path = f"{space}/startup.sh"
        try:
            with open(startup_path, 'w', encoding='utf-8') as f:
                f.write(startup_content)
            
            # Make executable
            os.chmod(startup_path, 0o755)
            print(f"✅ Created {space}/startup.sh")
            
        except Exception as e:
            print(f"❌ Failed to create startup script for {space}: {e}")

def update_requirements():
    """Update requirements with specific versions to avoid conflicts"""
    
    requirements_content = '''fastapi==0.104.1
uvicorn[standard]==0.24.0
torch>=2.0.0,<2.2.0
transformers>=4.35.0,<4.45.0
indictranstoolkit
PyPDF2==3.0.1
PyMuPDF==1.23.8
pytesseract==0.3.10
Pillow>=9.5.0,<11.0.0
reportlab==4.0.4
python-multipart==0.0.6
cython
sacremoses
sacrebleu
indic-nlp-library-itt
'''
    
    hf_spaces = [
        "hf_space_deploy",
        "hf_space_fix", 
        "hf_space_memory_fix",
        "hf_space_requirements_fix"
    ]
    
    for space in hf_spaces:
        requirements_path = f"{space}/requirements.txt"
        try:
            with open(requirements_path, 'w', encoding='utf-8') as f:
                f.write(requirements_content)
            print(f"✅ Updated {space}/requirements.txt")
        except Exception as e:
            print(f"❌ Failed to update requirements for {space}: {e}")

if __name__ == "__main__":
    print("🔧 Fixing HF Spaces cache permission issues...")
    print("=" * 60)
    
    # Fix cache permissions
    fix_cache_permissions()
    
    # Create startup scripts
    create_startup_script()
    
    # Update requirements with specific versions
    update_requirements()
    
    print("\n🎉 Cache permission fixes applied!")
    print("\n✅ Applied fixes:")
    print("- 🔧 Updated Dockerfiles with proper cache handling")
    print("- 📁 Added cache directory setup in main.py")
    print("- 🚀 Created startup scripts")
    print("- 📦 Updated requirements with stable versions")
    print("- 👤 Added non-root user for security")
    print("- 🔒 Set proper file permissions")
    
    print("\n🚀 Ready to redeploy!")
    print("Run: python deploy_all_hf_spaces.py")