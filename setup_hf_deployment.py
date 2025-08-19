#!/usr/bin/env python3
"""
Setup script for Hugging Face deployment
"""
import os
import subprocess
import json
from pathlib import Path

def check_hf_cli():
    """Check if Hugging Face CLI is installed"""
    try:
        result = subprocess.run("huggingface-hub --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Hugging Face CLI is installed")
            return True
    except:
        pass
    
    print("❌ Hugging Face CLI not found")
    print("📦 Install it with: pip install huggingface_hub")
    return False

def setup_hf_token():
    """Setup Hugging Face token"""
    print("\n🔑 Setting up Hugging Face token...")
    
    # Check if already logged in
    try:
        result = subprocess.run("huggingface-cli whoami", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Already logged in as: {result.stdout.strip()}")
            return True
    except:
        pass
    
    print("🔐 You need to login to Hugging Face")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'write' permissions")
    print("3. Copy the token")
    
    token = input("\nEnter your HF token: ").strip()
    
    if token:
        try:
            result = subprocess.run(f"huggingface-cli login --token {token}", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Successfully logged in to Hugging Face!")
                return True
            else:
                print(f"❌ Login failed: {result.stderr}")
        except Exception as e:
            print(f"❌ Error during login: {e}")
    
    return False

def create_hf_spaces():
    """Create HF spaces if they don't exist"""
    print("\n🏗️  Creating Hugging Face Spaces...")
    
    spaces = [
        {
            "name": "pdftransic-main",
            "title": "PDF Transic - Main",
            "description": "IndicTrans2 PDF Translation Service - Main deployment"
        },
        {
            "name": "pdftransic-memory-optimized", 
            "title": "PDF Transic - Memory Optimized",
            "description": "Memory-optimized version for large PDFs"
        },
        {
            "name": "pdftransic-enhanced",
            "title": "PDF Transic - Enhanced",
            "description": "Enhanced version with OCR and advanced features"
        },
        {
            "name": "pdftransic-stable",
            "title": "PDF Transic - Stable",
            "description": "Stable version with bug fixes"
        }
    ]
    
    created_spaces = []
    
    for space in spaces:
        print(f"\n📝 Creating space: {space['name']}")
        
        # Create space using HF CLI
        cmd = f"huggingface-cli repo create {space['name']} --type space --space_sdk docker"
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Created space: {space['name']}")
                created_spaces.append(space)
            else:
                if "already exists" in result.stderr.lower():
                    print(f"ℹ️  Space {space['name']} already exists")
                    created_spaces.append(space)
                else:
                    print(f"❌ Failed to create {space['name']}: {result.stderr}")
        except Exception as e:
            print(f"❌ Error creating {space['name']}: {e}")
    
    return created_spaces

def setup_git_remotes(created_spaces):
    """Setup git remotes for each space"""
    print("\n🔗 Setting up git remotes...")
    
    # Get HF username
    try:
        result = subprocess.run("huggingface-cli whoami", shell=True, capture_output=True, text=True)
        username = result.stdout.strip()
        print(f"👤 HF Username: {username}")
    except:
        username = input("Enter your HF username: ").strip()
    
    space_mapping = {
        "hf_space_deploy": "pdftransic-main",
        "hf_space_memory_fix": "pdftransic-memory-optimized", 
        "hf_space_fix": "pdftransic-enhanced",
        "hf_space_requirements_fix": "pdftransic-stable"
    }
    
    for local_dir, space_name in space_mapping.items():
        if Path(local_dir).exists():
            print(f"\n🔧 Setting up {local_dir} -> {space_name}")
            
            repo_url = f"https://huggingface.co/spaces/{username}/{space_name}"
            
            # Change to directory and setup remote
            original_cwd = os.getcwd()
            os.chdir(local_dir)
            
            try:
                # Initialize git if needed
                if not Path(".git").exists():
                    subprocess.run("git init", shell=True, capture_output=True)
                
                # Add remote
                subprocess.run("git remote remove origin", shell=True, capture_output=True)
                result = subprocess.run(f"git remote add origin {repo_url}", shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ Remote added: {repo_url}")
                else:
                    print(f"⚠️  Remote setup warning: {result.stderr}")
                
            except Exception as e:
                print(f"❌ Error setting up remote for {local_dir}: {e}")
            
            finally:
                os.chdir(original_cwd)

def main():
    """Main setup function"""
    print("🚀 Hugging Face Deployment Setup")
    print("=" * 50)
    
    # Check HF CLI
    if not check_hf_cli():
        return False
    
    # Setup HF token
    if not setup_hf_token():
        return False
    
    # Create spaces
    created_spaces = create_hf_spaces()
    
    # Setup git remotes
    setup_git_remotes(created_spaces)
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Run: python deploy_all_hf_spaces.py")
    print("2. Monitor your spaces at: https://huggingface.co/spaces")
    print("3. Test the deployed APIs")
    
    return True

if __name__ == "__main__":
    main()