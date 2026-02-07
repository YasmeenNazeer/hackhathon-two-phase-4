# Troubleshooting Guide: Infrastructure Issues

This guide addresses common infrastructure issues encountered when deploying the Todo Chatbot application to Kubernetes.

## Issue: Minikube Docker Driver Architecture Compatibility

### Problem
```
exec /usr/local/bin/entrypoint: exec format error: container exited unexpectedly
```

### Root Cause
This is typically an architecture mismatch issue between the Minikube base image and the host system, especially common on Windows systems with ARM/AMD architecture differences.

### Solutions

#### Solution 1: Clean Docker System
```bash
# Stop all containers
docker stop $(docker ps -aq)

# Clean up Docker system
docker system prune -a --volumes

# Restart Docker Desktop
```

#### Solution 2: Use Alternative Drivers
```bash
# For systems with Hyper-V enabled (requires admin privileges)
minikube delete
minikube start --driver=hyperv --hyperv-virtual-switch="Default Switch"

# For systems with VirtualBox installed
minikube delete
minikube start --driver=virtualbox
```

#### Solution 3: Adjust Memory and CPU Allocation
```bash
# Ensure Docker Desktop has adequate resources allocated
minikube delete
minikube start --driver=docker --memory=4096 --cpus=2
```

## Issue: Docker Authentication Required

### Problem
```
Sorry, you first need to sign in Docker Desktop
```

### Solution
1. Open Docker Desktop application
2. Sign in to your Docker account
3. Ensure Kubernetes is enabled in Docker Desktop settings
4. Restart Docker Desktop if needed

## Issue: Docker Desktop Kubernetes Not Enabled

### Problem
```
Unable to connect to the server: EOF
```

### Solution
1. Open Docker Desktop
2. Go to Settings > Kubernetes
3. Check "Enable Kubernetes"
4. Click Apply & Restart
5. Wait for Kubernetes to be fully started

## Issue: Insufficient System Resources

### Problem
```
Docker Desktop has only 3865MB memory but you specified 4096MB
```

### Solution
1. Open Docker Desktop Settings
2. Go to Resources
3. Increase memory allocation (recommended: 8GB+)
4. Increase CPU cores (recommended: 4+)
5. Apply and restart Docker Desktop

## Alternative Deployment Methods

If Minikube continues to have issues, consider these alternatives:

### Option 1: Kind (Kubernetes in Docker)
```bash
# Install kind
go install sigs.k8s.io/kind@v0.20.0

# Create cluster
kind create cluster --name todo-cluster

# Deploy application
kubectl config use-context kind-todo-cluster
# Then proceed with Helm deployment
```

### Option 2: MicroK8s (Linux only)
```bash
sudo snap install microk8s --classic
microk8s status --wait-ready
microk8s enable dns storage dashboard
# Then proceed with Helm deployment
```

### Option 3: Cloud Kubernetes (EKS, AKS, GKE)
For production deployments, consider managed Kubernetes services:
- Amazon EKS
- Azure AKS
- Google GKE

## Pre-flight Checks

Before attempting deployment, verify:

```bash
# Check Docker
docker --version
docker ps

# Check kubectl
kubectl version --client

# Check Helm
helm version

# Check cluster connectivity
kubectl cluster-info

# Check available resources
kubectl top nodes
```

## Quick Verification Commands

```bash
# Verify Docker images exist
docker images | grep -E "(backend|frontend|mcp-server)"

# Verify Helm chart is valid
helm lint ./helm/elevate-chatbot

# Verify cluster readiness
kubectl get nodes
kubectl get cs  # componentstatuses
```

## Recovery Steps

If deployment fails:

1. **Clean up**:
   ```bash
   helm list -A
   helm uninstall <release-name> -n <namespace>
   kubectl delete namespace <namespace>
   ```

2. **Reset cluster**:
   ```bash
   # For Minikube
   minikube delete

   # For Docker Desktop
   # Reset Kubernetes in Docker Desktop settings
   ```

3. **Retry deployment** after addressing identified issues.

## AI Tool Availability Check

Before using AI tools, verify they are installed:

```bash
# Check Docker AI
docker ai --help

# Check kubectl-ai
kubectl-ai --help

# Check Kagent
kagent --help
```

If any are missing, the scripts will fall back to standard tools automatically.