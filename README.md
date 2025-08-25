# Q-GridNavigator (Reinforcement Learning)

## Overview
This project implements a **custom GridWorld environment** and trains an agent using **Q-Learning** to achieve strategic navigation.  
The environment is inspired by real-world decision-making scenarios, where an agent (a “businessman”) must reach the **Goal (Success)** while avoiding **Hell states (Competitors)** and maximizing **Profit (Skills & Learnings)**.

The project demonstrates:
- **Custom Gym-style environment** with states, actions, and rewards  
- **Q-Learning implementation** for navigation and decision-making  
- Visualization of the Q-table and policy evolution over episodes  
- Comparative analysis of **different epsilon-decay setups** for convergence  

---

## Repository Structure

- `env_hac8488.py` : Custom environment implementation (states, rewards, actions)  
- `about.html` : Project description in HTML format  
- `grid.png` : Grid representation of the environment  
- `agent.png` : Visualization of the agent  
- `goal.png` : Goal state (Success)  
- `hell.png` : Hell states (Competitors)  
- `profit.png` : Reward/Profit states (Skills & Learnings)  

---

## Setup & Installation

### Prerequisites
- Python 3.8+  
- Required libraries:

pip install numpy matplotlib seaborn

## 📚 Main Libraries
NumPy – array operations, Q-table updates  
Matplotlib / Seaborn – visualization of Q-table and training results  

---

## 🔄 Workflow
1. Environment Setup  
   - Grid environment designed with states:  
     - Goal State → Success in Business  
     - Hell States → Competitors  
     - Profit States → Skills & Learnings  

2. Q-Learning Algorithm  
   - Q-table stores expected future rewards for (state, action) pairs  
   - Parameters:  
     - Learning Rate (α): Controls update weight of new info  
     - Discount Factor (γ): Balances immediate vs. future rewards  
     - Exploration Rate (ε): Decays to shift from exploration → exploitation  

3. Training  
   - Agent interacts with environment for a number of episodes  
   - Q-values updated iteratively until convergence  
   - Epsilon decay strategy tested with different values for stability  

4. Visualization  
   - Q-table heatmaps for learned state-action values  
   - Reward curves showing convergence rate across training setups  
   - Environment snapshots with agent, goal, and hell states  

---

## 📊 Results

**Initial Setup**  
- Epsilon Decay: 0.995  
- Epsilon Min: 0.1  
- Episodes: 5000  

**Modified Setup**  
- Epsilon Decay: 0.01  
- Epsilon Min: 0.3  
- Episodes: 10000  

### Key Findings
- Faster convergence observed with higher epsilon decay  
- Reduced total episodes required to learn optimal navigation  
- Agent successfully learned to avoid competitors and maximize profit  

---

## 🌟 Highlights
- Fully customizable RL environment following Gymnasium style  
- Demonstrates core Q-Learning principles with visual outputs  
- Side-by-side comparison of different hyperparameter setups  

---

## 🔮 Future Work
- Extend to Deep Q-Learning (DQN) for larger state spaces  
- Introduce multi-agent scenarios for cooperative/competitive learning  
- Integrate with OpenAI Gym for standardized benchmarking  
- Add real-time visualizations of agent’s learning process  

```bash