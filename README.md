# Safe-Reinforcement-Learning-using-Constraints
To study safety constraints in RL and their impact on performance using synthetic environments like GPSafety Gym and a Grid World Mars Explorer.
The below implements and compares **two Deep Q-Learning (DQN) agents**:
- An **unconstrained DQN agent** (no safety constraints)
- A **constrained DQN agent** (safety penalty for violations)

It runs both agents in the `SafetyPointGoal1-v0` environment (from Safety-Gymnasium) and compares their **reward** and **safety violations**.

---

### 1. **Setup and Imports**
  `pip install safety-gymnasium tensorflow mujoco pandas matplotlib`

- **Deep RL tools:**  
  `gymnasium` and `safety_gymnasium` — for the environment  
  `tensorflow.keras` — for building and training the neural network Q-function

- Seed setup to ensure reproducibility.

---

### 2. **Hyperparameters and Environment Details**
- `EPISODES`: Number of training episodes per agent.
- `GAMMA`: Discount factor for future rewards.
- `EPSILON_*`: For epsilon-greedy exploration.
- `BATCH_SIZE`, `MEMORY_SIZE`: For experience replay.
- `LEARNING_RATE`: For neural network optimizer.
- `LAMBDA_PENALTY`: Penalty to subtract from reward if a safety violation occurs (constrained agent only).
- `SIGN_SWITCH_EPISODE`: After this episode, the DQN update switches to using only the **sign** of the TD error (reduces variance).

#### **Action Discretization**
- The Safety-Gym environment's action space is continuous; we discretize it into `num_bins` per dimension (e.g., 3 bins → 9 actions for 2D).

---

## 3. **Helper Functions and Classes**
- **`index_to_action(index)`**: Converts a discrete action index to a continuous action vector using the defined bins.
- **`select_action(q_values)`**: Selects the greedy action (highest Q-value).
- **`build_model()`**: Builds a simple feed-forward neural network to estimate Q-values.  
  - Input: state observation  
  - Output: Q-value for each discrete action

- **`ReplayBuffer` class**: Implements the experience replay buffer for sampling minibatches for DQN updates.

---

## 4. **`train_agent(constrained=True)` Function**
Main function to train **one agent** (constrained or unconstrained).

### **Episode Loop**
- For each episode:
  1. **Reset the environment** and initialize counters.
  2. For each step:
     - **Action selection:** Epsilon-greedy: either random or model-predicted best.
     - **Environment step:** Take the action, get `next_obs, reward, cost, terminated, truncated, info`.
     - **Safety check:** If `cost > 0`, count a violation.
     - **Reward adjustment (constrained):** If `constrained` and violation occurs, subtract `LAMBDA_PENALTY` from reward.
     - **Experience replay:** Store transition in buffer.
     - **Batch training:** If enough samples, sample a minibatch and perform a DQN update:
       - **Before `SIGN_SWITCH_EPISODE`:** Standard DQN (use full TD error)
       - **After `SIGN_SWITCH_EPISODE`:** Only apply the sign of the TD error (variance reduction, slower convergence)
  3. **Update epsilon** (for exploration).
  4. **Log total reward and violations per episode.**

---

## 5. **Training Both Agents and Visualization**
- Run `train_agent(constrained=False)` for the unconstrained agent.
- Run `train_agent(constrained=True)` for the constrained agent.
- Log training times.

### **Plotting**
- **Left plot:** Total reward per episode (for both agents)
- **Right plot:** Safety violations per episode (for both agents)
---

## 6. **Key Features**
- **Comparison of safety tradeoffs:** Constrained vs unconstrained RL agents.
- **Incorporation of sign-based updates:** After a certain episode, the Q-value update uses only the sign of the increment (as a variance-reduction technique).
---
**Summary:**  
This code demonstrates how imposing safety constraints in RL can dramatically reduce unsafe actions, often with minimal impact on reward. It also enables experiment with update rules (standard/full increment vs. sign-only) for Q-learning

