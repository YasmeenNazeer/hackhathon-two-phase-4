#!/bin/bash
# script/docker-ai-push.sh
# Docker AI-assisted push script for the Elevate Chatbot application

set -e

REGISTRY=${1:-"localhost:5000"}
TAG=${2:-"latest"}

echo "Starting Docker AI-assisted push to $REGISTRY..."

# Tag and push images
echo "Tagging and pushing backend image..."
docker tag backend:latest $REGISTRY/backend:$TAG
if ! docker ai --help >/dev/null 2>&1; then
    echo "Docker AI not available. Using standard docker push..."
    docker push $REGISTRY/backend:$TAG
else
    echo "Using Docker AI for optimized push..."
    docker ai "recommend the most efficient way to push backend image to $REGISTRY"
    docker push $REGISTRY/backend:$TAG
fi

echo "Tagging and pushing frontend image..."
docker tag frontend:latest $REGISTRY/frontend:$TAG
if ! docker ai --help >/dev/null 2>&1; then
    echo "Docker AI not available. Using standard docker push..."
    docker push $REGISTRY/frontend:$TAG
else
    echo "Using Docker AI for optimized push..."
    docker push $REGISTRY/frontend:$TAG
fi

echo "Tagging and pushing mcp-server image..."
docker tag mcp-server:latest $REGISTRY/mcp-server:$TAG
if ! docker ai --help >/dev/null 2>&1; then
    echo "Docker AI not available. Using standard docker push..."
    docker push $REGISTRY/mcp-server:$TAG
else
    echo "Using Docker AI for optimized push..."
    docker push $REGISTRY/mcp-server:$TAG
fi

echo "All images pushed successfully!"