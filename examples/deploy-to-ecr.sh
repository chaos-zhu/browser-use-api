#!/bin/bash

# Configuration
IMAGE_NAME="browser-n8n-local"
TAG="latest"

# Get AWS account ID and region
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)

# Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names "$IMAGE_NAME" 2>/dev/null || \
    aws ecr create-repository --repository-name "$IMAGE_NAME" --image-scanning-configuration scanOnPush=true

# Get login command and execute it
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# Build the Docker image
echo "Building Docker image..."

cd ..

docker buildx build --platform linux/amd64,linux/arm64 -t "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$IMAGE_NAME:$TAG" --push .

# Clean up old images
docker image prune -f

# Print success message
echo "Successfully pushed image to ECR: $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$IMAGE_NAME:$TAG"
