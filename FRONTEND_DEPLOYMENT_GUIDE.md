
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
