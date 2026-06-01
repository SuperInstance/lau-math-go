# LAU Math Go

Cloud/services layer for the LAU math framework. REST APIs, gRPC services, fleet orchestration, and Kubernetes deployment — the operational wrapper around the math.

## What Go Handles

- Serving Lau math as a microservice
- Fleet management API
- Agent lifecycle management (spawn/kill/monitor)
- Persistence & observability
- Horizontal scaling
- CRDT-based fleet merge

## Packages

| Package | Description |
|---------|-------------|
| `pkg/matrix` | Matrix operations: multiply, invert, eigenvalues, Laplacian construction (wraps gonum) |
| `pkg/laplacian` | Graph Laplacian, spectral gap, heat kernel, harmonic projection, Fiedler vector |
| `pkg/agent` | Agent lifecycle: Observe → Predict → Update → Act → Conserve with belief state matrices |
| `pkg/fleet` | Fleet management: register agents, distribute work, collect results, CRDT merge |
| `pkg/conservation` | Noether charge monitoring, conservation law verification, alerting |
| `pkg/service` | REST API + gRPC service definitions |
| `pkg/config` | Hardware profile config (Jetson/RTX/Cloud/Chapel), auto-detect, backend dispatch |

## Quick Start

```bash
# Build
go build -o lau-math-server .

# Run with defaults
./lau-math-server

# Run with specific profile
./lau-math-server -profile rtx -backend gpu -port 8080

# Available profiles: cpu, jetson, rtx, cloud, cloud-gpu, chapel, auto
# Available backends: cpu, gpu, tensor, chapel, auto
```

## REST API

### Create Agent
```bash
curl -X POST http://localhost:8080/agent/create \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "my-agent", "dimension": 4, "learning_rate": 0.01}'
```

### Feed Observation
```bash
curl -X POST http://localhost:8080/agent/my-agent/observe \
  -H "Content-Type: application/json" \
  -d '{"rows": 4, "cols": 4, "data": [0.9,0.03,0.03,0.04,0.03,0.9,0.03,0.04,0.03,0.03,0.9,0.04,0.04,0.04,0.04,0.88]}'
```

### Get Agent State
```bash
curl http://localhost:8080/agent/my-agent/state
```

### Fleet Status
```bash
curl http://localhost:8080/fleet/status
```

### CRDT Merge
```bash
curl -X POST http://localhost:8080/fleet/merge \
  -H "Content-Type: application/json" \
  -d '{"fleet_id": "remote-fleet", "agents": {...}}'
```

### Health Check
```bash
curl http://localhost:8080/health
```

## gRPC

The protobuf definition is in `lau.proto`. Generate Go code:

```bash
protoc --go_out=. --go-grpc_out=. lau.proto
```

## Hardware Dispatch Guide

The `pkg/config` package auto-detects hardware capabilities and dispatches to the optimal backend:

| Profile | Use Case | Backend |
|---------|----------|---------|
| `jetson` | Edge deployment (NVIDIA Jetson) | GPU + Tensor Cores |
| `rtx` | Workstation (RTX 3090/4090) | GPU + Tensor Cores |
| `cloud` | General cloud (CPU) | CPU |
| `cloud-gpu` | Cloud GPU (A100/H100) | GPU + Tensor Cores |
| `chapel` | HPC cluster (Chapel) | Chapel distributed |
| `cpu` | Development / testing | CPU |

Auto-detection checks:
- `LAU_HARDWARE_PROFILE` env var (highest priority)
- `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES` (GPU detection)
- `JETSON_MODEL` (Jetson detection)
- CPU core count (heuristic for cloud vs HPC)

```bash
# Force a specific profile
export LAU_HARDWARE_PROFILE=rtx
./lau-math-server
```

## Docker

```bash
docker build -t lau-math-go .
docker run -p 8080:8080 -p 9090:9090 lau-math-go
```

## Kubernetes

```bash
kubectl apply -f k8s.yaml
```

Scales horizontally. Each pod runs its own fleet; use the CRDT merge API to synchronize across pods.

## Testing

```bash
# Run all tests
go test ./... -v

# Run with coverage
go test ./... -cover
```

## Architecture

```
                    ┌──────────────┐
                    │   REST API   │  :8080
                    │   gRPC API   │  :9090
                    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    │   service    │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────┴─────┐ ┌───┴───┐ ┌─────┴─────┐
        │   fleet   │ │ agent │ │conservation│
        └─────┬─────┘ └───┬───┘ └───────────┘
              │            │
        ┌─────┴────────────┴──────┐
        │       matrix/laplacian  │
        │     (gonum-backed)      │
        └─────────────────────────┘
```

## License

MIT
