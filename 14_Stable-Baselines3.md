# Stable-Baselines3

## 1. What is Stable-Baselines3?

**Stable-Baselines3** is a Python library for reinforcement learning (RL).

It provides reliable implementations of popular RL algorithms.

> **Simple definition:** Stable-Baselines3 lets an agent learn actions through rewards in an environment.

---

## 2. Basic Information

| Item | Details |
|---|---|
| **Name** | Stable-Baselines3 |
| **Type** | Reinforcement learning library |
| **Built with** | PyTorch |
| **Best for** | RL experiments and baselines |
| **Common environments** | Gymnasium/Gym-style environments |
| **Common algorithms** | PPO, DQN, A2C, SAC, TD3 |

---

# 3. What Problem Does It Solve?

In supervised learning, the model learns from correct answers.

In reinforcement learning, an agent learns by trying actions and receiving rewards.

```text
Agent chooses action
        |
        v
Environment changes
        |
        v
Agent receives reward
        |
        v
Agent improves policy
```

Examples:

- Game playing
- Robot control
- Trading simulations
- Resource allocation
- Navigation

---

# 4. RL Main Concepts

| Concept | Meaning |
|---|---|
| **Agent** | Learner/decision maker |
| **Environment** | World where agent acts |
| **State/Observation** | What the agent sees |
| **Action** | What the agent does |
| **Reward** | Feedback signal |
| **Policy** | Agent's strategy |
| **Episode** | One full run in environment |

---

# 5. Basic PPO Example

```python
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs, info = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
```

---

# 6. Common Algorithms

| Algorithm | Simple meaning | Best for |
|---|---|---|
| **PPO** | Stable policy optimization | General-purpose RL |
| **DQN** | Q-learning with neural networks | Discrete actions |
| **A2C** | Actor-critic method | Simple/fast experiments |
| **SAC** | Entropy-based actor-critic | Continuous control |
| **TD3** | Improved deterministic policy gradients | Continuous control |

---

# 7. Training Flow

```text
Create environment
        |
Choose RL algorithm
        |
Train agent
        |
Evaluate behavior
        |
Save model
        |
Use model
```

Save and load:

```python
model.save("ppo_cartpole")

loaded_model = PPO.load("ppo_cartpole", env=env)
```

---

# 8. Environment Requirement

Stable-Baselines3 expects Gym-style environments.

Basic environment API:

```text
reset() -> initial observation
step(action) -> observation, reward, done flags, info
```

The environment defines:

```text
observation_space
action_space
reward logic
episode ending logic
```

---

# 9. Advantages

- Reliable RL algorithm implementations
- Good for experiments
- Built on PyTorch
- Works with Gym-style environments
- Easy model saving/loading
- Good baselines for research and learning

---

# 10. Disadvantages

- RL is harder than supervised learning
- Training can be slow and unstable
- Reward design is difficult
- Needs simulation environments
- Not for normal regression/classification tasks

---

# 11. When to Use Stable-Baselines3

Use it when:

```text
You need reinforcement learning
You have an environment with rewards
You need PPO, DQN, A2C, SAC, or TD3
You are experimenting with agents
```

Avoid it when:

```text
You have labeled tabular data
You need ordinary classification/regression
You do not have an environment or reward signal
```

---

# 12. Quick Revision Table

| Topic | Meaning |
|---|---|
| **Stable-Baselines3** | RL algorithm library |
| **Agent** | Learner |
| **Environment** | World/task |
| **Reward** | Feedback signal |
| **Policy** | Strategy |
| **PPO** | General-purpose RL algorithm |
| **DQN** | Discrete-action RL algorithm |

---

# 13. Final Summary

```text
Stable-Baselines3 is for reinforcement learning.

Use it when:
    - An agent must learn actions
    - Rewards guide learning
    - You have a Gym-style environment

It is not for ordinary supervised ML.
```
