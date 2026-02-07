#!/bin/bash

# Check the status of the Elevate application deployment

echo "Checking status of Elevate application in namespace elevate-app..."

echo ""
echo "=== Pods ==="
kubectl get pods -n elevate-app

echo ""
echo "=== Services ==="
kubectl get svc -n elevate-app

echo ""
echo "=== Deployments ==="
kubectl get deployments -n elevate-app

echo ""
echo "=== Persistent Volume Claims ==="
kubectl get pvc -n elevate-app

echo ""
echo "=== Logs for Backend (last 20 lines) ==="
kubectl logs -l app=backend -n elevate-app --tail=20

echo ""
echo "=== Logs for MCP Server (last 20 lines) ==="
kubectl logs -l app=mcp-server -n elevate-app --tail=20

echo ""
echo "=== Logs for PostgreSQL (last 20 lines) ==="
kubectl logs -l app=postgres -n elevate-app --tail=20