# Hugging Face Spaces Deployment Guide

This guide will help you deploy your PDFTransic backend to Hugging Face Spaces using Docker.

## Prerequisites

- Git installed on your system
- Hugging Face account with access to your space: `https://huggingface.co/spaces/Syedkaif29/PdfTransic`
- Git credentials configured for Hugging Face

## Files Created for Deployment

The following files have been created in your `Pyback1/` directory for Hugging Face Spaces:

1. **Dockerfile** - Docker configuration for the backend
2. **README.md** - Space documentation with metadata
3. **.dockerignore** - Files to exclude from Docker build
4. **requirements.txt** - Updated Python dependencies

## Deployment Options

### Option 1: Automated Deployment (Recommended)

Run the automated deployment script:

```bash
python deploy_to_hf.py
```

This script will:
- Clone your HF Space repository
- Copy all necessary backend files
- Commit and push changes to HF Spaces

### Option 2: Manual Deployment

1. **Clone your HF Space repository:**
```bash
git clone https://huggingface.co/spaces/Syedkaif29/PdfTransic
cd PdfTransic
```

2. **Copy backend files to the space directory:**
```bash
# Copy from Pyback1/ to the HF space root
cp ../Pyback1/main.py .
cp ../Pyback1/requirements.txt .
cp ../Pyback1/Dockerfile .
cp ../Pyback1/README.md .
cp ../Pyback1/.dockerignore .
```

3. **Commit and push:**
```bash
git add .
git commit -m "Deploy PDFTransic backend to HF Spaces"
git push
```

## Important Notes

### Port Configuration
- Hugging Face Spaces uses port **7860** by default
- The Dockerfile is configured to expose port 7860
- Your FastAPI app will automatically bind to this port

### Model Loading
- **First startup will take 5-10 minutes** as the IndicTrans2 model downloads
- The space will show "Building" then "Running" status
- Use the `/health` endpoint to check when models are loaded

### Memory Requirements
- The space needs at least **4GB RAM** for the IndicTrans2 model
- Consider upgrading to a paid tier if you encounter memory issues

### Dependencies
- Updated `requirements.txt` to use `indictranstoolkit` from PyPI instead of Git
- All necessary dependencies for IndicTransToolkit are included

## Testing Your Deployment

Once deployed, test these endpoints:

1. **Health Check:**
```
GET https://syedkaif29-pdftransic.hf.space/health
```

2. **API Documentation:**
```
https://syedkaif29-pdftransic.hf.space/docs
```

3. **Simple Translation:**
```bash
curl -X POST "https://syedkaif29-pdftransic.hf.space/translate-simple" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world", "target_language": "hin_Deva"}'
```

## Troubleshooting

### Build Failures
- Check the "Logs" tab in your HF Space for build errors
- Ensure all dependencies in `requirements.txt` are correct
- Verify Dockerfile syntax

### Runtime Issues
- Monitor the `/health` endpoint for model loading status
- Check space logs for memory or timeout issues
- Ensure CORS is properly configured for your frontend

### Model Loading Timeout
- The IndicTrans2 model is large (~800MB)
- Initial download and loading can take 10+ minutes
- Be patient during first startup

## Space Configuration

Your space is configured with:
- **SDK:** Docker
- **Port:** 7860
- **License:** MIT
- **Hardware:** CPU (upgrade to GPU for faster inference)

## Next Steps

After successful deployment:

1. **Update your frontend** to use the new HF Spaces URL
2. **Test all endpoints** thoroughly
3. **Consider upgrading to GPU** for faster translation
4. **Monitor usage** and performance

## Frontend Integration

Update your frontend's API base URL to:
```typescript
const API_BASE_URL = "https://syedkaif29-pdftransic.hf.space";
```

Your backend will be accessible at: `https://syedkaif29-pdftransic.hf.space`