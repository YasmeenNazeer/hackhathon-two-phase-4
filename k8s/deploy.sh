#!/bin/bash

# Elevate App Kubernetes Deployment Script

echo "Deploying Elevate Task Management System to Kubernetes..."

# Apply the namespace first
echo "Creating namespace..."
kubectl apply -f 01-namespace.yaml

# Wait a moment for namespace creation
sleep 2

# Apply the secrets
echo "Applying secrets..."
kubectl apply -f 07-secrets.yaml

# Apply the configmap
echo "Applying configmap..."
kubectl apply -f 08-configmap.yaml

# Apply the persistent volume claim
echo "Applying PVC..."
kubectl apply -f 03-pvc.yaml

# Apply the database
echo "Applying PostgreSQL..."
kubectl apply -f 02-postgres.yaml

# Wait for database to be ready
echo "Waiting for PostgreSQL to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n elevate-app --timeout=120s

# Apply the MCP server
echo "Applying MCP server..."
kubectl apply -f 04-mcp-server.yaml

# Apply the backend
echo "Applying backend..."
kubectl apply -f 05-backend.yaml

# Apply the frontend
echo "Applying frontend..."
kubectl apply -f 06-frontend.yaml

echo "Deployment complete!"

echo "Services deployed:"
kubectl get svc -n elevate-app

echo ""
echo "To access the application:"
echo "Frontend: kubectl port-forward svc/frontend-service -n elevate-app 3000:3000"
echo "Backend: kubectl port-forward svc/backend-service -n elevate-app 8000:8000"
echo "MCP Server: kubectl port-forward svc/mcp-server-service -n elevate-app 8001:8001"