# Persian-Incursion-RL

**Reinforcement Learning framework for the digital wargame _Persian Incursion_.**

This project implements a self-play + training loop using Monte Carlo Tree Search (MCTS) and a policy–value model to explore AI strategies in the _Persian Incursion_ simulation.  
It supports both **Google Colab** and **Docker GPU** execution environments.

---

## 📁 Project Structure
Persian-Incursion-RL/
│
├── run_dynamic.py # Self-play simulation driver
├── trainer.py # Policy/value model training script
├── reinforcement_learning.py # RL utilities and training logic
├── mcts_agent.py # MCTS agent implementation
├── game_engine.py # Core simulation engine
├── mechanics.py # Rules and mechanics layer
├── actions_ops.py # Action definitions and operations
├── rules_global.py # Global ruleset
├── seeds_turn1_50/ # Example Turn-1 seed JSONs
├── self_play_data/ # Generated self-play data
├── runs_out_selfplay/ # Simulation output logs
└── models/ # Trained model checkpoints
