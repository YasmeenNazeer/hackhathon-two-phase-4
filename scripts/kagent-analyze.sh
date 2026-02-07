#!/bin/bash
# script/kagent-analyze.sh
# Kagent assisted cluster analysis script for the Elevate Chatbot application

set -e

NAMESPACE=${1:-"elevate-app"}

echo "Starting Kagent assisted cluster analysis for $NAMESPACE..."

# Check if kagent is available
if ! command -v kagent &> /dev/null; then
    echo "Kagent not available. Performing basic analysis..."

    echo "=== Cluster Info ==="
    kubectl cluster-info

    echo -e "\n=== Node Resources ==="
    kubectl top nodes

    echo -e "\n=== Pod Resources in $NAMESPACE ==="
    kubectl top pods -n $NAMESPACE

    echo -e "\n=== Storage Classes ==="
    kubectl get storageclasses

    echo -e "\n=== Ingress Controllers ==="
    kubectl get pods -A -l app.kubernetes.io/name=ingress-nginx
else
    echo "Kagent detected. Using AI-assisted cluster analysis..."

    # Use Kagent to analyze cluster health
    kagent "analyze the overall health of the Kubernetes cluster"
    kubectl cluster-info

    kagent "evaluate resource utilization across all nodes"
    kubectl top nodes

    kagent "analyze resource consumption of pods in $NAMESPACE namespace"
    kubectl top pods -n $NAMESPACE

    kagent "check storage availability and performance in the cluster"
    kubectl get storageclasses

    kagent "identify any potential ingress or networking issues"
    kubectl get pods -A -l app.kubernetes.io/name=ingress-nginx

    kagent "provide recommendations for optimizing resource allocation in $NAMESPACE"
    kubectl describe ns $NAMESPACE

    echo "Cluster analysis completed with Kagent assistance!"
fi

echo "Analysis completed."