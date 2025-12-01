# Distributed RL Platform

## Architecture Overview

This platform implements distributed reinforcement learning using actor-learner architecture with Kafka as the message broker.

## Component Connections

### 1. Actor → Kafka → Learner (Experience Flow)

Actors generate environment experiences and stream them to Kafka:
- Actors run parallel environments, collecting (state, action, reward, next_state)
- Experiences pushed to Kafka topic: `experiences`
- Learners batch-consume experiences for training

### 2. Learner → Kafka → Actor (Model Update Flow)

Learners train and broadcast model weights:
- Learners train PPO/other RL algorithms on batched experiences
- Updated model weights pushed to Kafka topic: `model_updates`
- Actors consume weights to sync their local policy networks

### 3. Frontend → API (Control Interface)

Web interface for training control:
- Frontend sends control requests (start/stop, scale actors, configure hyperparams)
- API forwards commands to Kafka topic: `commands.control`
- API consumes metrics from Kafka (`metrics.training`, `metrics.actors`)
- Returns status/metrics via REST API and WebSocket

### 4. MCP Server (AI Agent Control Interface)

Parallel control interface for AI agents:
- AI assistants (like Claude) control training via MCP protocol tools
- Same functionality as API (start/stop, scale, configure)
- Sends commands to Kafka topic: `commands.control`
- Consumes metrics from Kafka for reporting

### 5. API/MCP → Kubernetes (Dynamic Scaling)

Both control interfaces scale pods:
- Scale actor deployment (0-16 replicas)
- Scale learner deployment (0-1 replicas)
- Update ConfigMaps for environment changes

### 6. Docker Orchestration

All services containerized:
- `Dockerfile.actor`: Actor service container
- `Dockerfile.api`: API service container
- `Dockerfile.learner`: Learner service container
- `docker-compose.yaml`: Orchestrates all services with Kafka

### 7. Kubernetes Deployment

K8s configs for scalability:
- `actor.yaml`: Actor pod deployment
- `api.yaml`: API service deployment
- `learner.yaml`: Learner pod deployment
- `kafka.yaml`: Kafka broker deployment
- `configmap.yaml`: Shared configuration

## Data Flow

```
Frontend → API ↘
                 ↘ commands.control
AI Agent → MCP  → Kafka ↔ Actors ↔ Learner (core RL loop)
                 ↗ metrics.training
             K8s (pod scaling)

Core RL Loop (isolated):
Actor → Kafka (experiences) → Learner → Train → Kafka (weights) → Actor
```

## Communication Protocol

- **Actor-Learner**: Kafka pub/sub (asynchronous, core training)
- **Control→Kafka**: Commands via `commands.control` topic
- **Kafka→Control**: Metrics via `metrics.training`, `metrics.actors` topics
- **Frontend-API**: HTTP REST + WebSocket (synchronous)
- **AI Agent-MCP**: MCP protocol (tool-based control)

## Key Files

- `rl_core/actor.py`: Environment interaction logic
- `rl_core/learner.py`: Training loop implementation
- `rl_core/agent.py`: Neural network policy/value functions
- `mcp_server/api.py`: REST API endpoints
- `mcp_server/server.py`: MCP protocol server
