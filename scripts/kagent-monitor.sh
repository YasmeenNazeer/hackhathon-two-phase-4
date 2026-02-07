#!/bin/bash
# script/kagent-monitor.sh
# Kagent assisted monitoring script for the Elevate Chatbot application

set -e

NAMESPACE=${1:-"elevate-app"}

echo "Starting Kagent assisted monitoring for $NAMESPACE..."

# Check if kagent is available
if ! command -v kagent &> /dev/null; then
    echo "Kagent not available. Setting up basic monitoring..."

    echo "=== Monitoring Setup Instructions ==="
    echo "1. Deploy Prometheus and Grafana for metrics collection"
    echo "2. Set up logging aggregation with ELK stack or similar"
    echo "3. Configure alerting rules for critical metrics"
    echo "4. Monitor pod health and resource usage regularly"

    # Show current status
    echo -e "\n=== Current Pod Status ==="
    kubectl get pods -n $NAMESPACE

    echo -e "\n=== Current Services ==="
    kubectl get services -n $NAMESPACE

    echo -e "\n=== Recent Events ==="
    kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp' | tail -10
else
    echo "Kagent detected. Setting up AI-assisted monitoring..."

    # Use Kagent to establish monitoring
    kagent "establish comprehensive monitoring for the $NAMESPACE namespace"

    kubectl get pods -n $NAMESPACE

    kubectl get services -n $NAMESPACE

    kagent "recommend essential metrics to monitor for each service in $NAMESPACE"
    kubectl describe deployments -n $NAMESPACE

    kagent "set up alerts for critical thresholds in $NAMESPACE"
    kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp' | tail -10

    kagent "create dashboard recommendations for visualizing application health"
    kubectl top pods -n $NAMESPACE

    kagent "analyze historical data patterns and predict potential issues in $NAMESPACE"

    echo "Monitoring setup completed with Kagent assistance!"
fi

echo "Monitoring initialization completed."