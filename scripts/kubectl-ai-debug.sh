#!/bin/bash
# script/kubectl-ai-debug.sh
# kubectl-ai assisted debugging script for the Elevate Chatbot application

set -e

NAMESPACE=${1:-"elevate-app"}
POD_NAME=${2:-""}

echo "Starting kubectl-ai assisted debugging..."

if [ -z "$POD_NAME" ]; then
    echo "No specific pod provided. Analyzing all pods in $NAMESPACE..."

    # Check if kubectl-ai is available
    if ! kubectl ai --help >/dev/null 2>&1; then
        echo "kubectl-ai not available. Using standard kubectl debug commands..."

        echo "=== Pod Status ==="
        kubectl get pods -n $NAMESPACE

        echo -e "\n=== Failed Pods Logs ==="
        for pod in $(kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running,status.phase!=Succeeded -o jsonpath='{.items[*].metadata.name}'); do
            echo "--- Logs for $pod ---"
            kubectl logs -n $NAMESPACE $pod --all-containers=true || true
        done

        echo -e "\n=== Events ==="
        kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp'
    else
        echo "kubectl-ai detected. Using AI-assisted debugging..."

        kubectl-ai "analyze the health of pods in $NAMESPACE namespace"
        kubectl get pods -n $NAMESPACE

        kubectl-ai "find any failing pods in $NAMESPACE and explain potential causes"
        for pod in $(kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running,status.phase!=Succeeded -o jsonpath='{.items[*].metadata.name}'); do
            kubectl-ai "explain why pod $pod might be failing and suggest fixes"
            echo "--- Logs for $pod ---"
            kubectl logs -n $NAMESPACE $pod --all-containers=true || true
        done

        kubectl-ai "show me recent events in $NAMESPACE that might indicate problems"
        kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp'
    fi
else
    echo "Debugging specific pod: $POD_NAME"

    if ! kubectl ai --help >/dev/null 2>&1; then
        echo "kubectl-ai not available. Using standard kubectl debug commands..."

        echo "=== Pod Description ==="
        kubectl describe pod $POD_NAME -n $NAMESPACE

        echo -e "\n=== Pod Logs ==="
        kubectl logs -n $NAMESPACE $POD_NAME --all-containers=true

        echo -e "\n=== Previous Logs (if available) ==="
        kubectl logs -n $NAMESPACE $POD_NAME --previous || true
    else
        echo "kubectl-ai detected. Using AI-assisted debugging for $POD_NAME..."

        kubectl-ai "describe the pod $POD_NAME in $NAMESPACE and identify any issues"
        kubectl describe pod $POD_NAME -n $NAMESPACE

        kubectl-ai "analyze the logs of pod $POD_NAME and explain any errors or warnings"
        kubectl logs -n $NAMESPACE $POD_NAME --all-containers=true

        kubectl-ai "show any previous logs for pod $POD_NAME that might indicate restart reasons"
        kubectl logs -n $NAMESPACE $POD_NAME --previous || true
    fi
fi

echo "Debugging completed."