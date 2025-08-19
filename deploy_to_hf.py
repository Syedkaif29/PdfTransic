#!/usr/bin/env python3
"""
Deployment script for Hugging Face Spaces
"""
import os
import shutil
import subprocess
import sys

def run_command(cmd, cwd=None):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running command: {cmd}")
            print(f"Error output: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception running command {cmd}: {e}")
        return False

def main():
    # Configuration
    hf_repo_url = "https://huggingface.co/spaces/Syedkaif29/PdfTransic"
    local_hf_dir = "hf_space_deploy"
    backend_dir = "Pyback1"
    
    print("🚀 Starting Hugging Face Spaces deployment...")
    
    # Step 1: Clone the HF Space repository
    print("📥 Cloning Hugging Face Space repository...")
    if os.path.exists(local_hf_dir):
        shutil.rmtree(local_hf_dir)
    
    if not run_command(f"git clone {hf_repo_url} {local_hf_dir}"):
        print("❌ Failed to clone HF repository")
        return False
    
    # Step 2: Copy backend files to HF space
    print("📁 Copying backend files...")
    
    # Files to copy from backend
    files_to_copy = [
        "main.py",
        "requirements.txt", 
        "Dockerfile",
        "README.md",
        ".dockerignore"
    ]
    
    for file in files_to_copy:
        src = os.path.join(backend_dir, file)
        dst = os.path.join(local_hf_dir, file)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"✅ Copied {file}")
        else:
            print(f"⚠️  Warning: {file} not found in {backend_dir}")
    
    # Step 3: Commit and push to HF Space
    print("📤 Committing and pushing to Hugging Face Space...")
    
    os.chdir(local_hf_dir)
    
    commands = [
        "git add .",
        'git commit -m "Deploy PDFTransic backend to HF Spaces"',
        "git push"
    ]
    
    for cmd in commands:
        if not run_command(cmd):
            print(f"❌ Failed to execute: {cmd}")
            return False
    
    print("🎉 Deployment completed successfully!")
    print(f"🌐 Your space should be available at: {hf_repo_url}")
    print("⏳ Note: It may take a few minutes for the space to build and start")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)