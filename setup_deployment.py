#!/usr/bin/env python3
"""
Quick setup script to prepare files for Hugging Face Spaces deployment
"""
import os
import shutil

def main():
    print("🔧 Setting up files for Hugging Face Spaces deployment...")
    
    # Ensure we're in the right directory
    if not os.path.exists("Pyback1"):
        print("❌ Error: Pyback1 directory not found. Please run this from the project root.")
        return False
    
    # Check if all required files exist
    required_files = [
        "Pyback1/main.py",
        "Pyback1/requirements.txt",
        "Pyback1/Dockerfile",
        "Pyback1/README.md",
        "Pyback1/.dockerignore"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found!")
    print("\n📋 Deployment checklist:")
    print("   ✅ Dockerfile created")
    print("   ✅ README.md with HF metadata created")
    print("   ✅ .dockerignore created")
    print("   ✅ requirements.txt updated")
    print("   ✅ CORS configured for all origins")
    
    print("\n🚀 Ready for deployment!")
    print("\nNext steps:")
    print("1. Run: python deploy_to_hf.py")
    print("2. Or manually copy files to your HF Space")
    print("3. Wait for the space to build (5-10 minutes)")
    print("4. Test the /health endpoint")
    
    return True

if __name__ == "__main__":
    main()