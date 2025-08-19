#!/bin/bash

# Create cache directories with proper permissions
mkdir -p ~/.cache/huggingface/transformers
mkdir -p ~/.cache/huggingface/datasets
mkdir -p ~/.cache/torch

# Set proper permissions
chmod -R 755 ~/.cache

# Export environment variables
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets
export TORCH_HOME=~/.cache/torch

echo "🔧 Cache directories set up successfully"
echo "📁 HF_HOME: $HF_HOME"
echo "📁 TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"

# Start the FastAPI application
exec uvicorn main:app --host 0.0.0.0 --port 7860