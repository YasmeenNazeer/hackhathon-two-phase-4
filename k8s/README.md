# Elevate Task Management System - Production Kubernetes Deployment

This directory contains production-ready Kubernetes manifests for deploying the Elevate Task Management System.

## 🏗️ Architecture

The application consists of five main components with proper security and monitoring:

1. **PostgreSQL Database**: Persistent storage with health checks and security context
2. **MCP Server**: Multi-Agent Communication Protocol server for AI tool orchestration
3. **Backend**: FastAPI application with health endpoints and proper resource limits
4. **Frontend**: Next.js application with production build and security context
5. **Network Policies**: Secure internal communication and egress rules

## ⚙️ Production Features

### Security
- **Security Contexts**: All pods run as non-root users
- **Resource Limits**: CPU and memory constraints to prevent resource exhaustion
- **Network Policies**: Restrict traffic between components
- **Secrets Management**: Secure storage for sensitive data

### Reliability
- **Health Checks**: Liveness and readiness probes for all services
- **Persistent Storage**: Data preservation with PVC
- **Proper Resource Requests**: Guaranteed resources for stable operation

### Observability
- **Health Endpoints**: Dedicated endpoints for health checking
- **Structured Logging**: Consistent labeling for easy monitoring

## 🛠️ Prerequisites

- Kubernetes v1.32+ cluster (Docker Desktop, cloud provider)
- `kubectl` configured to connect to your cluster
- No dependency on minikube-specific features

## 🚀 Deployment

### Quick Deploy

```bash
# Make the deploy script executable
chmod +x deploy.sh

# Run the deployment
./deploy.sh
```

### Manual Deploy

Apply the manifests in order:

```bash
# 1. Create namespace
kubectl apply -f 01-namespace.yaml

# 2. Apply configmap
kubectl apply -f 08-configmap.yaml

# 3. Apply secrets (update with your actual values first!)
kubectl apply -f 07-secrets.yaml

# 4. Apply PVC
kubectl apply -f 03-pvc.yaml

# 5. Apply database
kubectl apply -f 02-postgres.yaml

# 6. Wait for database to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n elevate-app --timeout=120s

# 7. Apply MCP server
kubectl apply -f 04-mcp-server.yaml

# 8. Apply backend
kubectl apply -f 05-backend.yaml

# 9. Apply frontend
kubectl apply -f 06-frontend.yaml

# 10. Apply network policies
kubectl apply -f 09-network-policy.yaml
```

## 🌐 Accessing the Application

After deployment, you can access the services:

### External Access
If using a cloud provider or Docker Desktop with LoadBalancer support:
```bash
kubectl get svc frontend-service -n elevate-app
# Use the EXTERNAL-IP to access the frontend
```

### Local Access
For local clusters, use port forwarding:
```bash
# Access frontend
kubectl port-forward svc/frontend-service -n elevate-app 3000:80

# Access backend API
kubectl port-forward svc/backend-service -n elevate-app 8000:8000

# Access MCP Server
kubectl port-forward svc/mcp-server-service -n elevate-app 8001:8001

# Access PostgreSQL (internal only)
kubectl port-forward svc/postgres-service -n elevate-app 5432:5432
```

Then navigate to:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- MCP Server: http://localhost:8001
- Health checks: http://localhost:8000/health, http://localhost:8001/health

## 🔧 Services

| Service | Port | Description | Health Check |
|---------|------|-------------|--------------|
| frontend-service | 80/3000 | Next.js frontend application | `/` |
| backend-service | 8000 | FastAPI backend API | `/health` |
| mcp-server-service | 8001 | MCP server for AI tool orchestration | `/health` |
| postgres-service | 5432 | PostgreSQL database | `pg_isready` |

## 🔐 Configuration

The application is configured using:

- **Secrets**: For sensitive data like API keys and passwords
- **ConfigMaps**: For non-sensitive configuration
- **Environment Variables**: Injected via ConfigMaps and Secrets
- **Security Contexts**: Pod and container security settings

## 🔐 Updating Secrets

To update the secrets with your actual values:

```bash
# Encode your actual API key
echo -n "your-actual-gemini-api-key" | base64

# Encode your actual JWT secret
echo -n "your-actual-jwt-secret" | base64

# Encode your PostgreSQL password
echo -n "your-actual-postgres-password" | base64

# Then update the 07-secrets.yaml file with the encoded values
```

## 🔍 Troubleshooting

To check the status of your deployments:

```bash
# Check all resources
kubectl get all -n elevate-app

# Check pods with status
kubectl get pods -n elevate-app -o wide

# Check services
kubectl get svc -n elevate-app

# Check logs for specific components
kubectl logs -l app=backend -n elevate-app --tail=50
kubectl logs -l app=mcp-server -n elevate-app --tail=50
kubectl logs -l app=postgres -n elevate-app --tail=50

# Check events for debugging
kubectl get events -n elevate-app --sort-by='.lastTimestamp'

# Check resource usage
kubectl top pods -n elevate-app
```

## 📊 Health Monitoring

Each component includes health checks:

- **Backend**: GET `/health` returns `{"status": "healthy", "component": "backend"}`
- **MCP Server**: GET `/health` returns `{"status": "healthy", "component": "mcp-server"}`
- **Frontend**: GET `/` returns the main page
- **PostgreSQL**: Built-in `pg_isready` command

## 🧹 Cleanup

To remove all resources:

```bash
kubectl delete namespace elevate-app
```

Or remove individual components:
```bash
kubectl delete -f 09-network-policy.yaml
kubectl delete -f 06-frontend.yaml
kubectl delete -f 05-backend.yaml
kubectl delete -f 04-mcp-server.yaml
kubectl delete -f 02-postgres.yaml
kubectl delete -f 03-pvc.yaml
kubectl delete -f 07-secrets.yaml
kubectl delete -f 08-configmap.yaml
kubectl delete -f 01-namespace.yaml
```

## 📋 Verification Steps

After deployment, verify the system is working:

1. **Check pod status**:
   ```bash
   kubectl get pods -n elevate-app
   # All pods should be Running
   ```

2. **Check service connectivity**:
   ```bash
   kubectl get svc -n elevate-app
   # Services should have ClusterIPs assigned
   ```

3. **Test health endpoints**:
   ```bash
   # Port forward to test health
   kubectl port-forward svc/backend-service -n elevate-app 8000:8000 &
   curl http://localhost:8000/health
   # Should return: {"status": "healthy", "component": "backend"}
   ```

4. **Verify resource utilization**:
   ```bash
   kubectl top pods -n elevate-app
   # Resources should be within defined limits
   ```