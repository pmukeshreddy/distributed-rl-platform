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

### 3. Frontend → API (Inference/Monitoring)

Web interface for interaction:
- Frontend (index.html) sends HTTP requests to API service
- API loads trained models and serves predictions
- Returns visualization data and metrics

### 4. MCP Server → API/Learners (Protocol Interface)

Model Context Protocol server:
- Provides standardized interface for model interactions
- Handles tool calls and context management
- Communicates with API and Learners for orchestration

### 5. Docker Orchestration

All services containerized:
- `Dockerfile.actor`: Actor service container
- `Dockerfile.api`: API service container
- `Dockerfile.learner`: Learner service container
- `docker-compose.yaml`: Orchestrates all services with Kafka

### 6. Kubernetes Deployment

K8s configs for scalability:
- `actor.yaml`: Actor pod deployment
- `api.yaml`: API service deployment
- `learner.yaml`: Learner pod deployment
- `kafka.yaml`: Kafka broker deployment
- `configmap.yaml`: Shared configuration

## Data Flow

```
Environment → Actor → Kafka (experiences) → Learner → Train → Kafka (weights) → Actor → Update Policy
                                                ↓
                                              API → Inference → Frontend
```

## Communication Protocol

- **Actor-Learner**: Kafka pub/sub (asynchronous)
- **API-Frontend**: HTTP REST (synchronous)
- **MCP-Services**: MCP protocol (tool-based)

## Key Files

- `rl_core/actor.py`: Environment interaction logic
- `rl_core/learner.py`: Training loop implementation
- `rl_core/agent.py`: Neural network policy/value functions
- `mcp_server/api.py`: REST API endpoints
- `mcp_server/server.py`: MCP protocol server
