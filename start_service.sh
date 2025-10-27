#!/bin/bash

# Motion Generation Service Startup Script

echo "🚀 Starting Motion Generation Service..."

# Check if we're in the right conda environment
if [[ "$CONDA_DEFAULT_ENV" != "momask-plus" ]]; then
    echo "⚠️  Please activate the momask-plus conda environment first:"
    echo "   conda activate momask-plus"
    exit 1
fi

# Install additional dependencies
echo "📦 Installing service dependencies..."
pip install -r requirements_service.txt

# Set environment variables (adjust these as needed)
export CUDA_VISIBLE_DEVICES=0
export PORT="${PORT:-8010}"

# MinIO Configuration
export MINIO_ENDPOINT="${MINIO_ENDPOINT:-objectstorageapi.usw.sealos.io}"
export MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-f4b9q33o}"
export MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-62fqp8dzfmjqdpsr}"
export MINIO_BUCKET="${MINIO_BUCKET:-f4b9q33o-test1}"
export MINIO_SECURE="${MINIO_SECURE:-true}"

# OpenAI API Configuration for prompt rewriting
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://aiping.cn/api/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-QC-217cea37129465eda2e5917675a6dcd7-50c5417245a22c9d44887d8574bc275e}"
export OPENAI_MODEL="${OPENAI_MODEL:-DeepSeek-R1-0528}"

echo "🔧 Service Configuration:"
echo "   Port: $PORT"
echo "   MinIO Endpoint: $MINIO_ENDPOINT"
echo "   MinIO Access Key: $MINIO_ACCESS_KEY"
echo "   MinIO Bucket: $MINIO_BUCKET"
echo "   MinIO Secure: $MINIO_SECURE"
echo "   OpenAI Base URL: $OPENAI_BASE_URL"
echo "   OpenAI Model: $OPENAI_MODEL"
echo "   (Make sure MinIO is running and accessible)"

# Start the service
echo "🌟 Starting FastAPI service on http://localhost:$PORT"
python motion_service.py