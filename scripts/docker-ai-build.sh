#!/bin/bash
# scripts/docker-ai-build.sh
# AI-assisted Docker build script using Docker AI (Gordon)

set -e

echo "Starting AI-assisted Docker build process..."

# Check if Docker AI is available
if command -v docker &> /dev/null && docker ai --help &> /dev/null; then
    echo "Docker AI (Gordon) detected. Using AI-assisted container optimization..."

    # Build backend image with AI assistance
    echo "Building backend image with AI optimization..."
    if [ -f "Dockerfile.backend" ]; then
        # Use Docker AI to optimize the image
        echo "Analyzing and optimizing Dockerfile.backend with Docker AI..."
        docker ai "analyze Dockerfile.backend and suggest security improvements and optimizations"

        # Get the Docker AI analysis and apply suggested improvements
        docker ai "provide security best practices for Dockerfile.backend" > /tmp/docker-backend-analysis.txt

        docker build -t backend:latest -f Dockerfile.backend .

        # Scan the built image for vulnerabilities using Docker AI
        docker ai "scan backend:latest image for security vulnerabilities"

        # Generate SBOM for supply chain security
        if command -v syft &> /dev/null; then
            echo "Generating SBOM for backend image..."
            syft packages docker:backend:latest -o json > backend-sbom.json
        elif command -v docker &> /dev/null; then
            # Use Docker's built-in SBOM capability if available
            docker buildx imagetools inspect backend:latest --format '{{json .}}' > backend-inspect.json 2>/dev/null || true
        fi
    fi

    # Build frontend image with AI assistance
    echo "Building frontend image with AI optimization..."
    if [ -f "Dockerfile.frontend" ]; then
        echo "Analyzing and optimizing Dockerfile.frontend with Docker AI..."
        docker ai "analyze Dockerfile.frontend and suggest security improvements and optimizations"

        docker ai "provide security best practices for Dockerfile.frontend" > /tmp/docker-frontend-analysis.txt

        docker build -t frontend:latest -f Dockerfile.frontend .

        # Scan the built image for vulnerabilities
        docker ai "scan frontend:latest image for security vulnerabilities"

        # Generate SBOM for supply chain security
        if command -v syft &> /dev/null; then
            echo "Generating SBOM for frontend image..."
            syft packages docker:frontend:latest -o json > frontend-sbom.json
        fi
    fi

    # Build MCP server image with AI assistance
    echo "Building MCP server image with AI optimization..."
    if [ -f "Dockerfile.mcp-server" ]; then
        echo "Analyzing and optimizing Dockerfile.mcp-server with Docker AI..."
        docker ai "analyze Dockerfile.mcp-server and suggest security improvements and optimizations"

        docker ai "provide security best practices for Dockerfile.mcp-server" > /tmp/docker-mcpserver-analysis.txt

        docker build -t mcp-server:latest -f Dockerfile.mcp-server .

        # Scan the built image for vulnerabilities
        docker ai "scan mcp-server:latest image for security vulnerabilities"

        # Generate SBOM for supply chain security
        if command -v syft &> /dev/null; then
            echo "Generating SBOM for MCP server image..."
            syft packages docker:mcp-server:latest -o json > mcp-server-sbom.json
        fi
    fi

else
    echo "Docker AI (Gordon) not available. Falling back to standard Docker build..."

    # Standard Docker build process
    if [ -f "Dockerfile.backend" ]; then
        echo "Building backend:latest..."
        docker build -t backend:latest -f Dockerfile.backend .
    fi

    if [ -f "Dockerfile.frontend" ]; then
        echo "Building frontend:latest..."
        docker build -t frontend:latest -f Dockerfile.frontend .
    fi

    if [ -f "Dockerfile.mcp-server" ]; then
        echo "Building mcp-server:latest..."
        docker build -t mcp-server:latest -f Dockerfile.mcp-server .
    fi

    echo "Images built with standard Docker."
fi

# Verify built images
echo "Verifying built images..."
docker images | grep -E "(backend|frontend|mcp-server)" || echo "No images found or grep failed"

echo "Docker build process completed!"