# Phase 4: Local Kubernetes Deployment with AI Tools

## Overview
This phase implements deployment of the Todo Chatbot application on a local Kubernetes cluster using Minikube and Helm Charts, with integration of AI tools including Docker AI (Gordon), kubectl-ai, and Kagent for AI-assisted operations.

## Features Implemented
- ✅ Helm Charts for the complete application deployment
- ✅ Docker AI (Gordon) integration for container operations
- ✅ kubectl-ai integration for Kubernetes operations
- ✅ Kagent integration for advanced cluster analysis and optimization
- ✅ Minikube deployment automation
- ✅ Complete deployment workflow script

## Directory Structure
```
├── helm/
│   └── elevate-chatbot/           # Helm chart for the application
│       ├── Chart.yaml             # Chart metadata
│       ├── values.yaml            # Default configuration values
│       ├── templates/             # Kubernetes resource templates
│       │   ├── _helpers.tpl       # Helper templates
│       │   ├── backend-deployment.yaml
│       │   ├── frontend-deployment.yaml
│       │   ├── mcp-server-deployment.yaml
│       │   ├── postgres-deployment.yaml
│       │   ├── pvc.yaml
│       │   ├── secrets.yaml
│       │   ├── configmap.yaml
│       │   ├── network-policy.yaml
│       │   └── namespace.yaml
│       └── README.md              # Chart documentation
├── scripts/                       # AI-assisted operation scripts
│   ├── docker-ai-build.sh         # Docker AI-assisted build
│   ├── docker-ai-push.sh          # Docker AI-assisted push
│   ├── kubectl-ai-deploy.sh       # kubectl-ai-assisted deployment
│   ├── kubectl-ai-scale.sh        # kubectl-ai-assisted scaling
│   ├── kubectl-ai-debug.sh        # kubectl-ai-assisted debugging
│   ├── kagent-analyze.sh          # Kagent-assisted cluster analysis
│   ├── kagent-optimize.sh         # Kagent-assisted optimization
│   ├── kagent-monitor.sh          # Kagent-assisted monitoring
│   ├── minikube-setup.sh          # Minikube setup
│   └── deploy-all.sh              # Complete deployment workflow
├── Dockerfile.backend             # Backend container
├── Dockerfile.frontend            # Frontend container
├── Dockerfile.mcp-server          # MCP server container
├── docker-compose.yml             # Docker Compose configuration
└── k8s/                           # Kubernetes manifests
```

## Setup Instructions

### Prerequisites
- Docker Desktop 4.53+ with Docker AI enabled
- kubectl-ai plugin installed
- Kagent installed
- Helm 3.x installed
- Minikube installed
- Git

### Installation Steps

1. **Clone the repository:**
```bash
git clone <repository-url>
cd <repository-directory>
```

2. **Navigate to the project directory and run the complete deployment:**
```bash
cd D:\hackhathon-2\phasee-4
./scripts/deploy-all.sh
```

### Manual Deployment Steps (Alternative)

If you prefer to deploy manually instead of using the automated script:

1. **Start Minikube:**
```bash
./scripts/minikube-setup.sh
```

2. **Build container images:**
```bash
./scripts/docker-ai-build.sh
```

3. **Deploy using Helm:**
```bash
helm install elevate-release ./helm/elevate-chatbot --namespace elevate-app --create-namespace
```

4. **Verify the deployment:**
```bash
kubectl get pods -n elevate-app
kubectl get services -n elevate-app
```

## AI Tools Integration

### Docker AI (Gordon)
Used for container optimization and building:
```bash
# Build images with AI assistance
./scripts/docker-ai-build.sh

# Push images with AI assistance
./scripts/docker-ai-push.sh [registry] [tag]
```

### kubectl-ai
Used for Kubernetes operations:
```bash
# Deploy with AI assistance
./scripts/kubectl-ai-deploy.sh [release-name] [namespace]

# Scale with AI assistance
./scripts/kubectl-ai-scale.sh [service] [replicas] [namespace]

# Debug with AI assistance
./scripts/kubectl-ai-debug.sh [namespace] [pod-name]
```

### Kagent
Used for cluster analysis and optimization:
```bash
# Analyze cluster health
./scripts/kagent-analyze.sh [namespace]

# Optimize resources
./scripts/kagent-optimize.sh [namespace]

# Set up monitoring
./scripts/kagent-monitor.sh [namespace]
```

## Helm Chart Configuration

The Helm chart supports various customization options through values.yaml:

```yaml
# Image configurations
image:
  backend:
    repository: backend
    tag: latest
  frontend:
    repository: frontend
    tag: latest
  mcpServer:
    repository: mcp-server
    tag: latest

# Resource allocations
resources:
  backend:
    limits:
      cpu: 500m
      memory: 1Gi
    requests:
      cpu: 100m
      memory: 256Mi

# Service configurations
service:
  backend:
    port: 8000
    targetPort: 8000
    type: ClusterIP
  frontend:
    port: 3000
    targetPort: 3000
    type: LoadBalancer
```

## Customization Options

### Using Custom Values
Create a custom-values.yaml file and deploy with:
```bash
helm install elevate-release ./helm/elevate-chatbot -f custom-values.yaml --namespace elevate-app --create-namespace
```

### Scaling Applications
Use the AI-assisted scaling script:
```bash
./scripts/kubectl-ai-scale.sh backend 3 elevate-app
./scripts/kubectl-ai-scale.sh frontend 2 elevate-app
```

## Verification

After deployment, verify that all components are running:

```bash
# Check all pods
kubectl get pods -n elevate-app

# Check all services
kubectl get services -n elevate-app

# Check deployment status
kubectl get deployments -n elevate-app

# Access the application
minikube service frontend-service -n elevate-app
```

## Troubleshooting

### Common Issues and Solutions

1. **If Docker AI is not available:**
   - The scripts will automatically fall back to standard Docker commands
   - Verify Docker Desktop version and AI features are enabled

2. **If kubectl-ai is not available:**
   - Scripts will use standard kubectl commands
   - Install kubectl-ai using: `kubectl krew install ai`

3. **If Kagent is not available:**
   - Scripts will provide basic functionality
   - Install Kagent according to official documentation

4. **Minikube issues:**
   - Ensure sufficient system resources (at least 8GB RAM recommended)
   - Reset Minikube if needed: `minikube delete` then restart

### Debugging with AI Assistance
```bash
# Run the AI-assisted debug script
./scripts/kubectl-ai-debug.sh elevate-app
```

## Cleanup

To remove the deployment:
```bash
# Uninstall the Helm release
helm uninstall elevate-release -n elevate-app

# Optionally delete the namespace
kubectl delete namespace elevate-app

# Stop Minikube
minikube stop
```

## Dependencies
- Docker Desktop 4.53+ with Docker AI enabled
- kubectl-ai plugin
- Kagent
- Helm 3.x
- Minikube
- Git