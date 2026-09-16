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

---

# 14. How Reinforcement Learning Is Different

Supervised learning has correct answers.

```text
Input -> correct label
```

Reinforcement learning has rewards.

```text
State -> action -> reward -> new state
```

The agent may not know immediately whether an action was good.

Example:

```text
In a game, a move may look bad now but help win later.
```

This delayed reward problem makes RL difficult.

---

# 15. Policy, Value, and Q-value

| Term | Meaning |
|---|---|
| **Policy** | Strategy for choosing actions |
| **Value function** | Expected future reward from a state |
| **Q-value** | Expected reward for taking an action in a state |

Simple examples:

```text
Policy:
    If pole leans left, move cart left.

Value:
    This state is good because future reward is likely high.

Q-value:
    Taking action A from state S may give high future reward.
```

---

# 16. On-policy vs Off-policy

| Type | Meaning | Examples |
|---|---|---|
| **On-policy** | Learns from current policy's behavior | PPO, A2C |
| **Off-policy** | Can learn from past experience | DQN, SAC, TD3 |

Beginner interpretation:

```text
On-policy:
    Learn from what the current agent does.

Off-policy:
    Learn from stored or previous experiences too.
```

---

# 17. Vectorized Environments

RL training can be slow if one environment runs at a time.

Vectorized environments run multiple copies.

```text
Env 1 ┐
Env 2 ├──> collect experience faster
Env 3 ┤
Env 4 ┘
```

Example:

```python
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

env = make_vec_env("CartPole-v1", n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)
```

---

# 18. Evaluation

Training reward can be noisy.

Evaluate the agent separately.

```python
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=10
)

print(mean_reward, std_reward)
```

Why:

```text
One episode may be lucky or unlucky.
Multiple episodes give a better estimate.
```

---

# 19. Common RL Problems

| Problem | Meaning | Possible fix |
|---|---|---|
| Sparse rewards | Agent rarely gets feedback | Shape reward carefully |
| Unstable learning | Score jumps a lot | Tune learning rate, use more timesteps |
| Bad exploration | Agent repeats poor behavior | Adjust exploration/settings |
| Reward hacking | Agent exploits reward loophole | Redesign reward |
| Slow training | Environment is expensive | Use vectorized environments |

---

# 20. Choosing Algorithms

```text
Discrete actions:
    DQN, PPO, A2C

Continuous actions:
    PPO, SAC, TD3

Good first choice:
    PPO

Sample-efficient continuous control:
    SAC
```

Simple beginner advice:

```text
Start with PPO.
If actions are continuous and PPO struggles, try SAC.
```

---

# 21. Practical Workflow

```text
1. Define environment
2. Check observation/action spaces
3. Choose algorithm
4. Train for enough timesteps
5. Evaluate over many episodes
6. Tune hyperparameters
7. Save model
8. Test behavior visually if possible
```
