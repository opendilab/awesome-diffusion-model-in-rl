# Awesome Diffusion Model in RL

This is a collection of research papers for **Diffusion Model in RL**.
And the repository will be continuously updated to track the frontier of Diffusion RL.

Welcome to follow and star!

## Table of Contents

- [A Taxonomy of Diffusion Model in RL Algorithms](#a-taxonomy-of-diffusion-rl-algorithms)
- [Papers](#papers)

  - [Arxiv](#arxiv)
  - [ICLR 2023](#iclr-2023)(**<font color="red">New!!!</font>**) 
  - [ICRA 2023](#icra-2023)
  - [Neurips 2022](#neurips-2022)
  - [ICML 2022](#icml-2022)
- [Contributing](#contributing)

## Overview of Diffusion Model in RL

The Diffusion Model in RL was introduced by “Planning with Diffusion for Flexible Behavior Synthesis” by Janner, Michael, et al. It casts trajectory optimization as a **diffusion probabilistic model** that plans by iteratively refining trajectories.

![image info](./diffuser.png)

There is another way:  "Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning" by Wang, Z. proposed Diffusion Model as policy-optimization in offline RL, et al. Specifically, Diffusion-QL forms policy as a conditional diffusion model with states as the condition from the offline policy-optimization perspective.


![image info](./diffusion.png)

### Advantage

1. Bypass the need for bootstrapping for long term credit assignment.
2. Avoid undesirable short-sighted behaviors due to the discounting future rewards.
3. Enjoy the diffusion models widely used in language and vision, which are easy to scale and adapt to multi-modal data.

## Papers

```
format:
- [title](paper link) [links]
  - author1, author2, and author3...
  - publisher
  - key 
  - code 
  - experiment environment
```

### Arxiv

- [Diffusion-based Generation, Optimization, and Planning in 3D Scenes](https://arxiv.org/abs/2301.06015)
  - Siyuan Huang, Zan Wang, Puhao Li, Baoxiong Jia, Tengyu Liu, Yixin Zhu, Wei Liang, Song-Chun Zhu
  - Key: 3D Scene Understanding, Optimization, Planning
  - Code: [official](https://github.com/scenediffuser/Scene-Diffuser)
  - ExpEnv: [ScanNet](http://www.scan-net.org/), [MultiDex](https://github.com/tengyu-liu/GenDexGrasp), [PROX](https://prox.is.tue.mpg.de/index.html)

- [AdaptDiffuser: Diffusion Models as Adaptive Self-evolving Planners](https://arxiv.org/abs/2302.01877)
  - Zhixuan Liang, Yao Mu, Mingyu Ding, Fei Ni, Masayoshi Tomizuka, Ping Luo
  - Key: Planning, Generalizability
  - ExpEnv: Maze2D, MuJoCo, KUKA Robot


### ICLR 2023

- [Is Conditional Generative Modeling all you need for Decision-Making?](https://arxiv.org/abs/2211.15657)
  - Anurag Ajay, Yilun Du, Abhi Gupta, Joshua Tenenbaum, Tommi Jaakkola, Pulkit Agrawal
  - Publisher: ICLR 2023
  - Key: Offline RL, Generative Model, Policy Optimization
  - Code: [official](https://github.com/anuragajay/decision-diffuser/tree/main/code)
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)


- [Imitating Human Behaviour with Diffusion Models](https://arxiv.org/abs/2301.10677)
  - Tim Pearce, Tabish Rashid, Anssi Kanervisto, Dave Bignell, Mingfei Sun, Raluca Georgescu, Sergio Valcarcel Macua, Shan Zheng Tan, Ida Momennejad, Katja Hofmann, Sam Devlin
  - Publisher: ICLR 2023
  - Key: Offline RL, Policy Optimization, Imitation Learning
  - ExpEnv: Claw, Kitchen, CSGO

### ICRA 2023

- [Guided Conditional Diffusion for Controllable Traffic Simulation](https://arxiv.org/abs/2210.17366)
  - Ziyuan Zhong, Davis Rempe, Danfei Xu, Yuxiao Chen, Sushant Veer, Tong Che, Baishakhi Ray, Marco Pavone
  - Publisher: ICRA 2023
  - Key: Traffic Simulation, Multi-Agent 
  - ExpEnv: [nuScenes](https://github.com/nutonomy/nuscenes-devkit)

### Neurips 2022

- [Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning](https://arxiv.org/abs/2208.06193)
  - Zhendong Wang, Jonathan J Hunt, Mingyuan Zhou
  - Publisher:  Neurips Deep RL Workshop 2022
  - Key: Offline RL, Policy Optimization
  - Code: [official](https://github.com/zhendong-wang/diffusion-policies-for-offline-rl), [unofficial](https://github.com/twitter/diffusion-rl)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl)

### ICML 2022

- [Planning with Diffusion for Flexible Behavior Synthesis](https://arxiv.org/abs/2205.09991)
  - Michael Janner, Yilun Du, Joshua B. Tenenbaum, Sergey Levine
  - Publisher:  ICML 2022 (long talk)
  - Key: Offline RL,  Model-based RL, Trajectory Optimization
  - Code: [official](https://github.com/jannerm/diffuser)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl)

## Contributing

Our purpose is to make this repo even better. If you are interested in contributing, please refer to [HERE](CONTRIBUTING.md) for instructions in contribution.

## License

Awesome Diffusion Model in RL is released under the Apache 2.0 license.
