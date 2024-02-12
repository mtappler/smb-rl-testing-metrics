# Analysis of Metrics for RL Testing
This repository contains code implementing learning-based testing of deep RL agents playing Super Mario Bros., where we track metrics related computed from neural network used by the RL agents. It accompanies the paper "Bridging the Gap Between Models in RL: Test Models vs. Neural Networks", submitted to the AMOST workshop 2024.

The implementation is an adaptation of previous on [Differential Safety Testing of Deep RL Agents](https://github.com/mtappler/dlbt-smb-rl). 
The deep RL code is based on the [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html) on training deep RL agent for Super Mario Bros. and we use the [gym-super-bros environment](https://pypi.org/project/gym-super-mario-bros/).

# Structure
The dependencies required for setting up the experiments are given `environment.yml`, which can be used together with [Conda](https://docs.conda.io/en/latest/). The source code in the root directory contains the main implementation files and the directory `stats` includes some scripts to analyze and plot experimental results.

This is an initial version of our implementation, setup, and experiments.