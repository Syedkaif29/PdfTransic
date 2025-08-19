FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and set up cache directory
RUN useradd -m -u 1000 user && \
    mkdir -p /home/user/.cache/huggingface && \
    mkdir -p /home/user/.cache/torch && \
    chown -R user:user /home/user && \
    chmod -R 755 /home/user

# Set environment variables for Hugging Face
ENV HF_HOME=/home/user/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/user/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/home/user/.cache/huggingface/datasets
ENV TORCH_HOME=/home/user/.cache/torch
ENV HOME=/home/user

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make startup script executable
RUN chmod +x startup.sh

# Change ownership of the app directory
RUN chown -R user:user /app

# Switch to non-root user
USER user

# Expose port
EXPOSE 7860

# Use startup script
CMD ["./startup.sh"]