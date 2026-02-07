#!/bin/bash
# script/kubectl-ai-scale.sh
# kubectl-ai assisted scaling script for the Elevate Chatbot application

set -e

SERVICE=${1:-"backend"}
REPLICAS=${2:-"3"}
NAMESPACE=${3:-"elevate-app"}

echo "Starting kubectl-ai assisted scaling for $SERVICE to $REPLICAS replicas..."

# Check if kubectl-ai is available
if ! kubectl ai --help >/dev/null 2>&1; then
    echo "kubectl-ai not available. Using standard kubectl scale..."

    kubectl scale deployment $SERVICE-deployment --replicas=$REPLICAS -n $NAMESPACE
    echo "Scaling completed with standard kubectl."
else
    echo "kubectl-ai detected. Using AI-assisted scaling..."

    # Use kubectl-ai to suggest and perform scaling
    kubectl-ai "analyze the current resource usage of $SERVICE in $NAMESPACE and recommend appropriate scaling"
    kubectl-ai "scale the $SERVICE deployment in $NAMESPACE namespace to $REPLICAS replicas"

    kubectl scale deployment $SERVICE-deployment --replicas=$REPLICAS -n $NAMESPACE

    echo "AI-assisted scaling completed!"
fi

echo "Verifying scaling results..."
kubectl get pods -n $NAMESPACE -l app=$SERVICE

echo "Scaling verification complete."