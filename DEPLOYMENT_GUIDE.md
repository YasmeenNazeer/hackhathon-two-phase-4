# Deployment Guide for Phase IV: Todo Chatbot on Kubernetes

This guide outlines how to deploy the enhanced Todo Chatbot application to a local Kubernetes cluster using the implemented enhancements.

## Prerequisites

1. **Docker Desktop** with Kubernetes enabled OR **Minikube** installed
2. **kubectl** - Kubernetes command-line tool
3. **Helm** - Package manager for Kubernetes
4. **Docker AI (Gordon)** - For AI-assisted container optimization (optional)
5. **kubectl-ai** - AI-assisted Kubernetes operations (optional)
6. **Kagent** - AI-assisted cluster analysis (optional)

## Infrastructure Setup

### Option A: Using Minikube
```bash
# Start Minikube cluster
minikube start --driver=docker

# Verify cluster is running
kubectl cluster-info
minikube status
```

### Option B: Using Docker Desktop Kubernetes
```bash
# Enable Kubernetes in Docker Desktop settings
# Verify cluster is running
kubectl cluster-info
```

## Building Container Images

The application consists of three services:
- **Backend**: Python FastAPI service with Todo API
- **Frontend**: Next.js web application
- **MCP Server**: Management and Control Plane server

```bash
# Build all images using the AI-assisted script
./scripts/docker-ai-build.sh

# Or build manually:
docker build -f Dockerfile.backend -t backend:latest .
docker build -f Dockerfile.frontend -t frontend:latest .
docker build -f Dockerfile.mcp-server -t mcp-server:latest .
```

## Deploying with Enhanced Helm Charts

### 1. Verify Helm Installation
```bash
helm version
helm repo update
```

### 2. Deploy the Application

#### For Development Environment:
```bash
helm upgrade --install elevate-chatbot-dev ./helm/elevate-chatbot \
  --namespace elevate-dev --create-namespace \
  -f ./helm/elevate-chatbot/values-dev.yaml
```

#### For Staging Environment:
```bash
helm upgrade --install elevate-chatbot-staging ./helm/elevate-chatbot \
  --namespace elevate-staging --create-namespace \
  -f ./helm/elevate-chatbot/values-staging.yaml
```

#### For Production Environment:
```bash
helm upgrade --install elevate-chatbot-prod ./helm/elevate-chatbot \
  --namespace elevate-prod --create-namespace \
  -f ./helm/elevate-chatbot/values-prod.yaml
```

### 3. Complete Deployment Process
Use the main deployment script which incorporates all AI tools:

```bash
# Deploy with default namespace (elevate-app)
./scripts/deploy-all.sh

# Or deploy with custom namespace and release name
./scripts/deploy-all.sh my-namespace my-release
```

## Enhanced Features Deployed

### Security Enhancements
- **Runtime Security**: Read-only root filesystems, non-root users
- **Network Policies**: Granular ingress/egress controls
- **Pod Security Policies**: Privilege escalation prevention
- **External Secrets**: Vault/AWS KMS integration capability

### Observability Stack
- **Prometheus**: Metrics collection and monitoring
- **Jaeger**: Distributed tracing
- **Service Monitors**: Automatic endpoint discovery
- **Structured Logging**: Enhanced logging capabilities

### AI Tool Integration
- **Docker AI (Gordon)**: Container optimization and security scanning
- **kubectl-ai**: Intelligent Kubernetes operations
- **Kagent**: Cluster analysis and optimization recommendations

## Verification Steps

After deployment, verify the application:

```bash
# Check all pods are running
kubectl get pods -n <your-namespace>

# Check all services are available
kubectl get services -n <your-namespace>

# Check deployments
kubectl get deployments -n <your-namespace>

# Check logs for any issues
kubectl logs -l app=backend -n <your-namespace>
kubectl logs -l app=frontend -n <your-namespace>
kubectl logs -l app=mcp-server -n <your-namespace>

# Port forward to access the application
kubectl port-forward -n <your-namespace> svc/frontend-service 3000:3000
```

## Accessing the Application

The application can be accessed via:
- **Frontend**: Port 3000 (LoadBalancer service)
- **Backend**: Port 8000 (ClusterIP service)
- **MCP Server**: Port 8080 (ClusterIP service)

For Minikube:
```bash
minikube service frontend-service -n <your-namespace>
```

For Docker Desktop:
```bash
kubectl port-forward -n <your-namespace> svc/frontend-service 3000:3000
# Then access http://localhost:3000
```

## AI-Assisted Operations

### Cluster Analysis with Kagent
```bash
./scripts/kagent-analyze.sh <your-namespace>
./scripts/kagent-optimize.sh <your-namespace>
```

### Intelligent Troubleshooting with kubectl-ai
```bash
./scripts/kubectl-ai-debug.sh <your-namespace>
```

### Scaling with AI Assistance
```bash
./scripts/kubectl-ai-scale.sh <your-namespace>
```

## Multi-Environment Configuration

The application supports three environments with different configurations:

- **Development**: Lower resource requests, debug mode, basic monitoring
- **Staging**: Higher replicas, standard resources, full monitoring
- **Production**: High availability, external secrets, enhanced security

## Troubleshooting

If you encounter issues:

1. **Check if all prerequisites are met**
2. **Verify cluster status**: `kubectl cluster-info`
3. **Check for resource constraints**: `kubectl top nodes`
4. **Review pod statuses**: `kubectl get pods -A`
5. **Use AI tools for intelligent debugging**: `./scripts/kubectl-ai-debug.sh`

## Cleanup

To remove the deployment:
```bash
helm uninstall <release-name> -n <your-namespace>
kubectl delete namespace <your-namespace>
```

For Minikube, to stop the cluster:
```bash
minikube stop
```

## Summary

The deployment includes all Phase IV enhancements:
✅ Removed inline code from Kubernetes ConfigMaps
✅ Enhanced security with multi-stage builds and runtime security
✅ Implemented external secrets management
✅ Added comprehensive network policies
✅ Integrated observability stack (Prometheus, Jaeger)
✅ Multi-environment configuration support
✅ Full AI tool integration (Docker AI, kubectl-ai, Kagent)