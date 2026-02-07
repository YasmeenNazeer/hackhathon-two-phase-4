#!/bin/bash
# script/kagent-optimize.sh
# Kagent assisted resource optimization script for the Elevate Chatbot application

set -e

NAMESPACE=${1:-"elevate-app"}

echo "Starting Kagent assisted resource optimization for $NAMESPACE..."

# Check if kagent is available
if ! command -v kagent &> /dev/null; then
    echo "Kagent not available. Providing basic optimization suggestions..."

    echo "=== Current Resource Requests/Limits ==="
    kubectl get deployments -n $NAMESPACE -o=jsonpath='{range .items[*]}{.metadata.name}{"\n"}{range .spec.template.spec.containers[*]}  Requests: {.resources.requests}{"\n"}  Limits: {.resources.limits}{"\n"}{end}{"\n"}{end}'

    echo -e "\n=== Resource Utilization ==="
    kubectl top pods -n $NAMESPACE

    echo -e "\nSuggested optimizations:"
    echo "- Review resource requests and limits based on actual usage"
    echo "- Consider enabling horizontal pod autoscaling"
    echo "- Optimize image sizes and layer caching"
else
    echo "Kagent detected. Using AI-assisted resource optimization..."

    # Use Kagent to analyze and suggest optimizations
    kagent "analyze current resource utilization in $NAMESPACE and identify optimization opportunities"
    kubectl top pods -n $NAMESPACE

    kagent "evaluate the current resource requests and limits for deployments in $NAMESPACE"
    kubectl get deployments -n $NAMESPACE -o=jsonpath='{range .items[*]}{.metadata.name}{"\n"}{range .spec.template.spec.containers[*]}  Requests: {.resources.requests}{"\n"}  Limits: {.resources.limits}{"\n"}{end}{"\n"}{end}'

    kagent "recommend optimized resource allocations for all deployments in $NAMESPACE based on observed usage"
    kubectl describe deployments -n $NAMESPACE

    kagent "suggest implementation of Horizontal Pod Autoscaler for deployments in $NAMESPACE"
    echo "Consider creating HPAs for dynamic scaling based on CPU/memory usage."

    kagent "analyze the efficiency of container images and suggest improvements"
    kubectl get pods -n $NAMESPACE -o=jsonpath='{range .items[*]}{.spec.containers[*].image}{"\n"}{end}' | sort | uniq -c

    kagent "generate specific commands to apply resource optimizations to deployments in $NAMESPACE"

    echo "Resource optimization recommendations completed with Kagent assistance!"
fi

echo "Optimization analysis completed."