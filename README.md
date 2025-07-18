# microgrids
Adaptive Control in Smart Microgrids using Deep Reinforcement Learning

This project demonstrates the application of reinforcement learning (RL) to control a simplified smart microgrid environment. Inspired by real-world challenges in industrial automation and energy efficiency, the agent learns how to balance 01) energy demand, 02) battery storage, and 03) renewable generation through sequential decision-making.


 --> simulates a microgrid environment with changing energy inputs (solar, wind, battery) and usage demands. The RL agent learns a policy to balance storage, consumption, and external supply, optimising for energy efficiency and cost over time.

 this makes use of **proximal policy optimisation** to:
 01. reduce reliance on external grid power 
 02. minimise operational costs 
 03. maintain energy balance + battery health 

 ## files 
 - smart_microgrid_env.py = custom OpenAI gym-compatible env't for microgrid
 - ppo_agent.py = proixmal policy optimisation agent w/ PyTorch
 - train.py = training script for RL agent
 - requirements.txt = dependencies for project

## to start ! 
install dependencies:
```bash
pip install -r requirements.txt
