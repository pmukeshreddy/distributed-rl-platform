# distributed-rl-platform

/distributed-rl-platform
├── rl_core/
│   ├── agent.py          # PPO agent (actor-critic network)
│   ├── env_wrapper.py    # Gym env with serialization
│   ├── replay_buffer.py  # Prioritized experience replay
│   └── trainer.py        # Training loop
├── kafka_layer/
│   ├── producer.py       # Actor → experiences to topic
│   ├── consumer.py       # Learner ← batches from topic
│   └── schemas.py        # Avro/JSON schemas for experiences
├── mcp_server/
│   ├── server.py         # MCP protocol handler
│   ├── tools.py          # start_training, get_metrics, set_hyperparam
│   └── resources.py      # Training state, model checkpoints
├── k8s/
│   ├── actor-deployment.yaml
│   ├── learner-deployment.yaml
│   ├── kafka-statefulset.yaml
│   └── mcp-service.yaml
├── docker/
│   ├── Dockerfile.actor
│   ├── Dockerfile.learner
│   └── Dockerfile.mcp
└── configs/
    └── training_config.yaml


