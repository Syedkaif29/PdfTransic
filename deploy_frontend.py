#!/usr/bin/env python3
"""
Frontend deployment script for Vercel/Netlify
"""
import os
import subprocess
import json

def run_command(cmd, cwd=None):
    """Run a shell command"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_dependencies():
    """Check if required tools are installed"""
    print("🔍 Checking dependencies...")
    
    # Check Node.js
    success, stdout, stderr = run_command("node --version")
    if success:
        print(f"✅ Node.js: {stdout.strip()}")
    else:
        print("❌ Node.js not found. Please install Node.js 16+")
        return False
    
    # Check npm
    success, stdout, stderr = run_command("npm --version")
    if success:
        print(f"✅ npm: {stdout.strip()}")
    else:
        print("❌ npm not found")
        return False
    
    return True

def setup_frontend():
    """Set up frontend for deployment"""
    frontend_dir = "Pyfornt/transfornt"
    
    if not os.path.exists(frontend_dir):
        print(f"❌ Frontend directory not found: {frontend_dir}")
        return False
    
    print("📦 Installing frontend dependencies...")
    success, stdout, stderr = run_command("npm install", cwd=frontend_dir)
    if not success:
        print(f"❌ Failed to install dependencies: {stderr}")
        return False
    
    print("✅ Dependencies installed successfully")
    
    # Test build
    print("🔨 Testing build...")
    success, stdout, stderr = run_command("npm run build", cwd=frontend_dir)
    if not success:
        print(f"❌ Build failed: {stderr}")
        return False
    
    print("✅ Build successful")
    return True

def create_deployment_guide():
    """Create deployment instructions"""
    guide = """
# 🚀 Frontend Deployment Guide

## Environment Setup

Your frontend is now configured with environment-based API URLs:

- **Local Development**: Uses `http://localhost:8000`
- **Production**: Uses environment variable `VITE_API_BASE_URL`

## Vercel Deployment

1. **Install Vercel CLI** (if not already installed):
   ```bash
   npm install -g vercel
   ```

2. **Deploy to Vercel**:
   ```bash
   cd Pyfornt/transfornt
   vercel
   ```

3. **Set Environment Variable** in Vercel dashboard:
   - Go to your project settings
   - Add environment variable:
     - Name: `VITE_API_BASE_URL`
     - Value: `https://syedkaif29-pdftransic.hf.space`

## Netlify Deployment

1. **Install Netlify CLI** (if not already installed):
   ```bash
   npm install -g netlify-cli
   ```

2. **Deploy to Netlify**:
   ```bash
   cd Pyfornt/transfornt
   netlify deploy --prod
   ```

3. **Set Environment Variable** in Netlify dashboard:
   - Go to Site settings > Environment variables
   - Add: `VITE_API_BASE_URL` = `https://syedkaif29-pdftransic.hf.space`

## GitHub Integration (Recommended)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Add frontend deployment configuration"
   git push
   ```

2. **Connect to Vercel/Netlify**:
   - Import your GitHub repository
   - Set the build directory to `Pyfornt/transfornt`
   - Add environment variable: `VITE_API_BASE_URL`

## Local Development

To run locally:
```bash
cd Pyfornt/transfornt
npm run dev
```

The app will use `http://localhost:8000` for API calls.

## Testing

After deployment, test these URLs:
- Your deployed frontend URL
- API health check from frontend
- Translation functionality
- PDF upload functionality
"""
    
    with open("FRONTEND_DEPLOYMENT_GUIDE.md", "w", encoding="utf-8") as f:
        f.write(guide)
    
    print("📖 Created FRONTEND_DEPLOYMENT_GUIDE.md")

def main():
    print("🚀 Setting up frontend for deployment...")
    
    if not check_dependencies():
        return False
    
    if not setup_frontend():
        return False
    
    create_deployment_guide()
    
    print("\n🎉 Frontend setup complete!")
    print("\n📋 Next steps:")
    print("1. Read FRONTEND_DEPLOYMENT_GUIDE.md")
    print("2. Choose Vercel or Netlify for deployment")
    print("3. Set VITE_API_BASE_URL environment variable")
    print("4. Deploy and test!")
    
    print(f"\n🔗 Your backend API: https://syedkaif29-pdftransic.hf.space")
    print("🔗 Frontend will be deployed to your chosen platform")
    
    return True

if __name__ == "__main__":
    main()