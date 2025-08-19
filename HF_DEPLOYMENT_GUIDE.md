# 🚀 Hugging Face Spaces Deployment Guide

Your PDF translation spaces are ready to deploy! Follow this guide to get them live on Hugging Face.

## 📋 Prerequisites

1. **Hugging Face Account**: Create account at [huggingface.co](https://huggingface.co)
2. **Git**: Ensure git is installed and configured
3. **HF CLI**: Install with `pip install huggingface_hub`

## 🔧 Quick Setup

### Option 1: Automated Setup (Recommended)
```bash
# Run the setup script
python setup_hf_deployment.py

# Then deploy all spaces
python deploy_all_hf_spaces.py
```

### Option 2: Manual Setup

1. **Install HF CLI**:
   ```bash
   pip install huggingface_hub
   ```

2. **Login to HF**:
   ```bash
   huggingface-cli login
   ```
   - Get your token from: https://huggingface.co/settings/tokens
   - Create token with "write" permissions

3. **Create Spaces** (via HF website):
   - Go to https://huggingface.co/new-space
   - Create these spaces:
     - `your-username/pdftransic-main`
     - `your-username/pdftransic-memory-optimized`
     - `your-username/pdftransic-enhanced`
     - `your-username/pdftransic-stable`
   - Set SDK to "Docker" for all spaces

## 🚀 Deployment Options

### Deploy All Spaces
```bash
python deploy_all_hf_spaces.py
```
Choose option 1 to deploy all spaces at once.

### Deploy Specific Spaces
```bash
python deploy_all_hf_spaces.py
```
Choose option 2 and select which spaces to deploy.

### Deploy with Confirmation
```bash
python deploy_all_hf_spaces.py
```
Choose option 3 to confirm each space before deployment.

## 📁 Space Mapping

| Local Directory | HF Space Name | Purpose |
|----------------|---------------|---------|
| `hf_space_deploy` | `pdftransic-main` | Main production deployment |
| `hf_space_memory_fix` | `pdftransic-memory-optimized` | Memory-optimized for large PDFs |
| `hf_space_fix` | `pdftransic-enhanced` | Enhanced features with OCR |
| `hf_space_requirements_fix` | `pdftransic-stable` | Stable version with fixes |

## 🔍 What Gets Deployed

Each space includes:
- ✅ **main.py** - FastAPI application with all endpoints
- ✅ **requirements.txt** - Python dependencies
- ✅ **Dockerfile** - Container configuration with OCR support
- ✅ **README.md** - Comprehensive documentation
- ✅ **Memory optimizations** - 60-80% memory reduction
- ✅ **Enhanced PDF processing** - Multiple extraction methods
- ✅ **OCR support** - For scanned PDFs
- ✅ **Smart chunking** - For large documents
- ✅ **PDF generation** - Download translated PDFs

## 🌐 API Endpoints

Once deployed, each space will have these endpoints:

### Core Translation
- `POST /translate-simple` - Simple text translation
- `POST /translate` - Batch translation
- `POST /translate-pdf` - PDF translation
- `POST /translate-pdf-enhanced` - Advanced PDF with download

### Monitoring
- `GET /health` - Health check
- `GET /memory-info` - Memory usage
- `POST /clear-memory` - Clear GPU cache
- `GET /languages` - Supported languages

### Documentation
- `GET /docs` - Interactive API docs
- `GET /` - API information

## 🔧 Troubleshooting

### Common Issues

1. **Git not configured**:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

2. **HF token issues**:
   - Ensure token has "write" permissions
   - Re-login: `huggingface-cli login`

3. **Space already exists**:
   - The script will use existing spaces
   - Or create new ones with different names

4. **Build failures**:
   - Check space logs on HF dashboard
   - Verify Dockerfile and requirements.txt
   - Memory issues: Use memory-optimized version

### Manual Deployment

If automated deployment fails, you can deploy manually:

```bash
# For each space directory
cd hf_space_deploy
git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
git add .
git commit -m "Deploy PDF translation service"
git push origin main
```

## 📊 Monitoring Deployment

1. **Check HF Dashboard**: https://huggingface.co/spaces
2. **Monitor Build Logs**: Click on your space → "Logs" tab
3. **Test API**: Use the `/docs` endpoint once built
4. **Memory Usage**: Monitor via `/memory-info` endpoint

## 🎯 Testing Your Deployment

Once deployed, test with:

```bash
# Health check
curl https://YOUR_USERNAME-SPACE_NAME.hf.space/health

# Simple translation
curl -X POST "https://YOUR_USERNAME-SPACE_NAME.hf.space/translate-simple" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world", "target_language": "hin_Deva"}'

# PDF translation
curl -X POST "https://YOUR_USERNAME-SPACE_NAME.hf.space/translate-pdf" \
     -F "file=@test.pdf" \
     -F "target_language=hin_Deva"
```

## 🚀 Next Steps

After successful deployment:

1. **Update README**: Add your space URLs
2. **Configure Settings**: Adjust space visibility/settings
3. **Monitor Performance**: Check logs and memory usage
4. **Share**: Your translation API is now live!

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review HF Spaces documentation
3. Check space build logs for specific errors

---

🎉 **Ready to deploy?** Run `python setup_hf_deployment.py` to get started!