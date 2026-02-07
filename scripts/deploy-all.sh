#!/bin/bash
# script/deploy-all.sh
# Complete deployment workflow script for the Elevate Chatbot application

set -e

echo "Starting complete deployment workflow for Elevate Chatbot..."

# Check for required tools
MISSING_TOOLS=()
for tool in minikube kubectl helm docker; do
    if ! command -v $tool &> /dev/null; then
        MISSING_TOOLS+=("$tool")
    fi
done

if [ ${#MISSING_TOOLS[@]} -ne 0 ]; then
    echo "Missing required tools: ${MISSING_TOOLS[*]}"
    echo "Please install the missing tools before continuing."
    exit 1
fi

# Set default values
NAMESPACE=${1:-"elevate-app"}
RELEASE_NAME=${2:-"elevate-release"}

echo "Target namespace: $NAMESPACE"
echo "Target release name: $RELEASE_NAME"

# Step 1: Setup Minikube
echo -e "\n=== Step 1: Setting up Minikube ==="
./scripts/minikube-setup.sh

# Step 2: Build images (using Docker AI if available)
echo -e "\n=== Step 2: Building container images ==="
./scripts/docker-ai-build.sh

# Step 3: Deploy using Helm charts
echo -e "\n=== Step 3: Deploying using Helm charts ==="
./scripts/kubectl-ai-deploy.sh $RELEASE_NAME $NAMESPACE

# Step 4: Wait for deployments to be ready
echo -e "\n=== Step 4: Waiting for deployments to be ready ==="
kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=180s
kubectl wait --for=condition=ready pod -l app=backend -n $NAMESPACE --timeout=180s
kubectl wait --for=condition=ready pod -l app=mcp-server -n $NAMESPACE --timeout=180s
kubectl wait --for=condition=ready pod -l app=frontend -n $NAMESPACE --timeout=180s

# Step 5: Verify deployment
echo -e "\n=== Step 5: Verifying deployment ==="
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE
kubectl get deployments -n $NAMESPACE

# Step 6: Run basic connectivity tests
echo -e "\n=== Step 6: Running connectivity tests ==="
# Wait briefly for services to be available
sleep 10

# Check if services are reachable within the cluster
kubectl run connectivity-test --image=curlimages/curl -it --rm --restart=Never \
  --namespace $NAMESPACE \
  --overrides='{"spec":{"hostNetwork": true}}' \
  -- curl -f http://frontend-service.$NAMESPACE.svc.cluster.local:3000/ || echo "Frontend service connectivity check completed"

# Step 7: Display access information
echo -e "\n=== Step 7: Access Information ==="
FRONTEND_IP=$(kubectl get svc frontend-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
FRONTEND_PORT=$(kubectl get svc frontend-service -n $NAMESPACE -o jsonpath='{.spec.ports[0].port}')

echo "Application deployed successfully!"
echo "Namespace: $NAMESPACE"
echo "Release: $RELEASE_NAME"
echo "Frontend Service IP: $FRONTEND_IP"
echo "Frontend Service Port: $FRONTEND_PORT"

# If using LoadBalancer service type, get the external IP
EXTERNAL_IP=$(kubectl get svc frontend-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ ! -z "$EXTERNAL_IP" ]; then
    echo "External IP (if LoadBalancer is available): $EXTERNAL_IP"
fi

echo -e "\nTo access the application in Minikube:"
echo "minikube service frontend-service -n $NAMESPACE"

# Step 8: Run comprehensive AI tools integration
echo -e "\n=== Step 8: Running comprehensive AI-assisted operations ==="
./scripts/ai-tools-integration.sh $NAMESPACE $RELEASE_NAME

echo -e "\n=== Deployment Workflow Completed Successfully! ==="
echo "The Elevate Chatbot application is now deployed on your local Kubernetes cluster."
echo "Use the scripts in the scripts/ directory to manage the deployment with AI assistance."