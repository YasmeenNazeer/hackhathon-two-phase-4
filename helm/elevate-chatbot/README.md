# Elevate Chatbot Helm Chart

A Helm chart for deploying the Elevate Chatbot application on Kubernetes.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+

## Installing the Chart

To install the chart with the release name `elevate-release`:

```bash
helm install elevate-release ./helm/elevate-chatbot --namespace elevate-app --create-namespace
```

## Uninstalling the Chart

To uninstall/delete the `elevate-release` deployment:

```bash
helm uninstall elevate-release -n elevate-app
```

## Configuration

The following table lists the configurable parameters of the Elevate Chatbot chart and their default values.

| Parameter                     | Description                                        | Default                             |
|-------------------------------|----------------------------------------------------|-------------------------------------|
| `global.imageRegistry`        | Global Docker image registry                       | `""`                                |
| `global.imagePullSecrets`     | Global Docker registry secret names as an array    | `[]`                                |
| `global.storageClass`         | Global storage class for dynamic provisioning      | `""`                                |
| `image.backend.repository`    | Backend image repository                           | `backend`                           |
| `image.backend.tag`           | Backend image tag                                  | `latest`                            |
| `image.backend.pullPolicy`    | Backend image pull policy                          | `IfNotPresent`                      |
| `image.frontend.repository`   | Frontend image repository                          | `frontend`                          |
| `image.frontend.tag`          | Frontend image tag                                 | `latest`                            |
| `image.frontend.pullPolicy`   | Frontend image pull policy                         | `IfNotPresent`                      |
| `image.mcpServer.repository`  | MCP Server image repository                        | `mcp-server`                        |
| `image.mcpServer.tag`         | MCP Server image tag                               | `latest`                            |
| `image.mcpServer.pullPolicy`  | MCP Server image pull policy                       | `IfNotPresent`                      |
| `image.postgres.repository`   | Postgres image repository                          | `postgres`                          |
| `image.postgres.tag`          | Postgres image tag                                 | `"15.3"`                            |
| `image.postgres.pullPolicy`   | Postgres image pull policy                         | `IfNotPresent`                      |
| `replicaCount.backend`        | Number of backend replicas                         | `1`                                 |
| `replicaCount.frontend`       | Number of frontend replicas                        | `1`                                 |
| `replicaCount.mcpServer`      | Number of MCP server replicas                      | `1`                                 |
| `replicaCount.postgres`       | Number of Postgres replicas                        | `1`                                 |
| `resources.backend.limits.cpu`    | Backend CPU limit                              | `500m`                              |
| `resources.backend.limits.memory` | Backend memory limit                           | `1Gi`                               |
| `resources.backend.requests.cpu`  | Backend CPU request                            | `100m`                              |
| `resources.backend.requests.memory`| Backend memory request                        | `256Mi`                             |
| `service.backend.port`        | Backend service port                               | `8000`                              |
| `service.backend.targetPort`  | Backend service target port                        | `8000`                              |
| `service.backend.type`        | Backend service type                               | `ClusterIP`                         |
| `service.frontend.port`       | Frontend service port                              | `3000`                              |
| `service.frontend.targetPort` | Frontend service target port                       | `3000`                              |
| `service.frontend.type`       | Frontend service type                              | `LoadBalancer`                      |
| `persistence.enabled`         | Enable persistence for Postgres                    | `true`                              |
| `persistence.storageSize`     | Size of persistent storage                         | `1Gi`                               |
| `secrets.postgresPassword`    | Password for Postgres                              | `password`                          |
| `secrets.jwtSecret`           | JWT secret for authentication                      | `your-secret-key`                   |

## Values

Specify each parameter using the `--set key=value[,key=value]` argument to `helm install`.

Alternatively, a YAML file that specifies the values for the parameters can be provided while installing the chart:

```bash
helm install elevate-release -f values.yaml ./helm/elevate-chatbot --namespace elevate-app --create-namespace
```