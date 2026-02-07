#!/bin/bash
# script/minikube-setup.sh
# Minikube setup script for the Elevate Chatbot application

set -e

echo "Setting up Minikube for Elevate Chatbot deployment..."

# Check if minikube is installed
if ! command -v minikube &> /dev/null; then
    echo "Minikube is not installed. Please install Minikube first."
    echo "Visit https://minikube.sigs.k8s.io/docs/start/ for installation instructions."
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Start minikube with appropriate resources
echo "Starting Minikube with sufficient resources..."
minikube start --cpus=2 --memory=3072 --disk-size=8g

# Enable required addons
echo "Enabling required Minikube addons..."
minikube addons enable ingress
minikube addons enable metrics-server
minikube addons enable dashboard

# Set the docker environment to point to minikube's docker daemon
echo "Configuring Docker environment for Minikube..."
eval $(minikube docker-env)

# Verify the setup
echo "Verifying Minikube setup..."
kubectl cluster-info
kubectl get nodes

# Wait for nodes to be ready
echo "Waiting for nodes to be ready..."
kubectl wait --for=condition=ready node --all --timeout=120s

echo "Minikube setup completed successfully!"
echo "Minikube dashboard URL: $(minikube dashboard url --format='{{.URL}}')"