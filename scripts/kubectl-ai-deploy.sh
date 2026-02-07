#!/bin/bash
# script/kubectl-ai-deploy.sh
# kubectl-ai assisted deployment script for the Elevate Chatbot application

set -e

RELEASE_NAME=${1:-"elevate-release"}
NAMESPACE=${2:-"elevate-app"}

echo "Starting kubectl-ai assisted deployment..."

# Check if kubectl-ai is available
if ! kubectl ai --help >/dev/null 2>&1; then
    echo "kubectl-ai not available. Using standard kubectl..."

    # Create namespace
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

    # Install Helm chart
    helm upgrade --install $RELEASE_NAME ./helm/elevate-chatbot --namespace $NAMESPACE --create-namespace

    echo "Deployment completed with standard kubectl."
else
    echo "kubectl-ai detected. Using AI-assisted deployment..."

    # Use kubectl-ai to assist with deployment
    kubectl-ai "create namespace $NAMESPACE if it doesn't exist"
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

    echo "Using kubectl-ai to deploy the Elevate Chatbot application..."
    kubectl-ai "deploy the Helm chart located at ./helm/elevate-chatbot with release name $RELEASE_NAME in namespace $NAMESPACE"
    helm upgrade --install $RELEASE_NAME ./helm/elevate-chatbot --namespace $NAMESPACE --create-namespace

    echo "AI-assisted deployment completed!"
fi

echo "Checking deployment status..."
kubectl-ai "show the status of pods in $NAMESPACE namespace" 2>/dev/null || kubectl get pods -n $NAMESPACE

echo "Deployment verification complete."