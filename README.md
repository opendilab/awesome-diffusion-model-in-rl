# Awesome Diffusion Model in RL
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) [![docs](https://img.shields.io/badge/docs-latest-blue)](https://github.com/opendilab/awesome-diffusion-model-in-rl) ![visitor badge](https://visitor-badge.lithub.cc/badge?page_id=opendilab.awesome-diffusion-model-in-rl&left_text=Visitors)  ![GitHub stars](https://img.shields.io/github/stars/opendilab/awesome-diffusion-model-in-rl?color=yellow) ![GitHub forks](https://img.shields.io/github/forks/opendilab/awesome-diffusion-model-in-rl?color=9cf) [![GitHub license](https://img.shields.io/github/license/opendilab/awesome-diffusion-model-in-rl)](https://github.com/opendilab/awesome-diffusion-model-in-rl/blob/main/LICENSE)


This is a collection of research papers for **Diffusion Model in RL**.
And the repository will be continuously updated to track the frontier of Diffusion RL.

Welcome to follow and star!

## Table of Contents

- [Awesome Diffusion Model in RL](#awesome-diffusion-model-in-rl)
  - [Table of Contents](#table-of-contents)
  - [Overview of Diffusion Model in RL](#overview-of-diffusion-model-in-rl)
    - [Advantage](#advantage)
  - [Papers](#papers)
    - [Arxiv](#arxiv)
    - [ICML 2024](#icml-2024)(**<font color="red">New!!!</font>**) 
    - [ICLR 2024](#iclr-2024)
    - [CVPR 2024](#cvpr-2024)
    - [ICML 2023](#icml-2023)
    - [ICLR 2023](#iclr-2023)
    - [ICRA 2023](#icra-2023)
    - [NeurIPS 2022](#neurips-2022)
    - [ICML 2022](#icml-2022)
  - [Contributing](#contributing)
  - [License](#license)

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

- [Enhancing Deep Reinforcement Learning: A Tutorial on Generative Diffusion Models in Network Optimization](https://arxiv.org/abs/2308.05384)
  - Hongyang Du, Ruichen Zhang, Yinqiu Liu, Jiacheng Wang, Yijing Lin, Zonghang Li, Dusit Niyato, Jiawen Kang, Zehui Xiong, Shuguang Cui, Bo Ai, Haibo Zhou, Dong In Kim
  - Key: Generative Diffusion Models, Incentive Mechanism Design, Semantic Communications, Internet of Vehicles
  - ExpEnv: None

- [3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations](https://arxiv.org/abs/2403.03954)
  - Yanjie Ze, Gu Zhang, Kangning Zhang, Chenyuan Hu, Muhan Wang, Huazhe Xu
  - Key: 3D Diffusion Policy, visual imitation learning
  - ExpEnv: MetaWorld, Adroit, DexArt

- [Diffusion Actor-Critic: Formulating Constrained Policy Iteration as Diffusion Noise Regression for Offline Reinforcement Learning](https://arxiv.org/abs/2405.20555)
  - Linjiajie Fang, Ruoxue Liu, Jing Zhang, Wenjia Wang, Bing-Yi Jing
  - Key: diffusion models, Actor-Critic, offline RL
  - ExpEnv: D4RL

- [NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration](https://openreview.net/pdf?id=FhQRJW71h5)
  - Ajay Sridhar, Dhruv Shah, Catherine Glossop, Sergey Levine
  - Key: diffusion models, Offline RL
  - ExpEnv: Real-world robot manipulation

- [IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies](https://arxiv.org/pdf/2304.10573)
  - Philippe Hansen-Estruch, Ilya Kostrikov, Michael Janner, Jakub Grudzien Kuba, Sergey Levine
  - Key: diffusion models, Offline RL
  - ExpEnv: D4RL

- [To the Noise and Back: Diffusion for Shared Autonomy](https://arxiv.org/pdf/2302.12244)
  - Takuma Yoneda, Luzhe Sun, and Ge Yang, Bradly Stadie, Matthew Walter
  - Key: diffusion models, Imitation, Robotics
  - ExpEnv:  2D Control, Lunar Lander, Lunar Reacher and Block Pushing

- [PlayFusion: Skill Acquisition via Diffusion from Language-Annotated Play](https://arxiv.org/pdf/2312.04549)
  - Lili Chen, Shikhar Bahl, Deepak Pathak
  - Key: diffusion models, Imitation, Robotics
  - ExpEnv: CALVIN, Franka Kitchen, Language-Conditioned Ravens

- [XSkill: Cross Embodiment Skill Discovery](https://arxiv.org/pdf/2307.09955)
  - Mengda Xu, Zhenjia Xu, Cheng Chi, Manuela Veloso, Shuran Song
  - Key: diffusion models, Imitation, Robotics
  - ExpEnv: Real-world robot manipulation

- [Diffusion Co-Policy for Synergistic Human-Robot Collaborative Tasks](https://arxiv.org/pdf/2305.12171)
  - Eley Ng, Ziang Liu, Monroe Kennedy III
  - Key: diffusion models, Human-in-the-loop, Robotics
  - ExpEnv: Human-in-the-Loop Simulation

- [GenAug: Retargeting behaviors to unseen situations via Generative Augmentation](https://arxiv.org/pdf/2302.06671)
  - Zoey Chen, Sho Kiami, Abhishek Gupta, Vikash Kumar
  - Key: diffusion models, Data Synthesizer, Robotics
  - ExpEnv: end-to-end vision manipulation tasks

- [Scaling Robot Learning with Semantically Imagined Experience](https://arxiv.org/pdf/2302.11550)
  - Tianhe Yu, Ted Xiao, Austin Stone, Jonathan Tompson, Anthony Brohan, Su Wang, Jaspiar Singh, Clayton Tan, Dee M, Jodilyn Peralta, Brian Ichter, Karol Hausman, Fei Xia
  - Key: diffusion models, Data Synthesizer, Robotics
  - ExpEnv: robot manipulation tasks

- [Synthetic Experience Replay](https://arxiv.org/pdf/2303.06614)
  - Cong Lu, Philip J. Ball, Yee Whye Teh, Jack Parker-Holder
  - Key: diffusion models, Data Synthesizer
  - ExpEnv: D4RL

- [Value function estimation using conditional diffusion models for control](https://arxiv.org/pdf/2306.07290)
  - Bogdan Mazoure, Walter Talbott, Miguel Angel Bautista, Devon Hjelm, Alexander Toshev, Josh Susskind
  - Key: diffusion models, off-policy learning, offline RL, reinforcement learning, robotics
  - ExpEnv: D4RL

- [Safe Offline Reinforcement Learning with Feasibility-Guided Diffusion Model](https://arxiv.org/abs/2401.10700)
  - Yinan Zheng, Jianxiong Li, Dongjie Yu, Yujie Yang, Shengbo Eben Li, Xianyuan Zhan, Jingjing Liu
  - Key: Time-independent classifier-guided, Safe offline RL
  - Code: [official](https://github.com/ZhengYinan-AIR/FISOR)
  - ExpEnv: DSRL

- [World Models via Policy-Guided Trajectory Diffusion](https://arxiv.org/abs/2312.08533)
  - Marc Rigter, Jun Yamada, Ingmar Posner
  - Key: world models, model-based RL, policy guidance
  - ExpEnv: Gym MuJoCo

- [Diffusion Models for Reinforcement Learning: A Survey](https://arxiv.org/abs/2311.01223)
  - Zhengbang Zhu, Hanye Zhao, Haoran He, Yichao Zhong, Shenyu Zhang, Yong Yu, Weinan Zhang
  - Key: survey

- [Boosting Continuous Control with Consistency Policy](https://arxiv.org/abs/2310.06343)
  - Yuhui Chen, Haoran Li, Dongbin Zhao
  - Key: Q-learning, sample efficiency, Consistency policy
  - ExpEnv: DMC, Gym MuJoCo, D4RL

- [DiffCPS: Diffusion Model based Constrained Policy Search for Offline Reinforcement Learning](https://arxiv.org/abs/2310.05333)
  - Longxiang He, Linrui Zhang, Junbo Tan, Xueqian Wang
  - Key: Constrained policy search, Offline-RL
  - ExpEnv: D4RL

- [Learning to Reach Goals via Diffusion](https://arxiv.org/abs/2310.02505)
  - Vineet Jain, Siamak Ravanbakhsh
  - Key: Constrained policy search, Offline-RL
  - ExpEnv: offline goal-conditioned setting

- [AlignDiff: Aligning Diverse Human Preferences via Behavior-Customisable Diffusion Model ](https://arxiv.org/abs/2310.02054)
  - Zibin Dong, Yifu Yuan, Jianye Hao, Fei Ni, Yao Mu, Yan Zheng, Yujing Hu, Tangjie Lv, Changjie Fan, Zhipeng Hu
  - Key: RLHF, Alignment, Classifier-free
  - ExpEnv: Gym MuJoCo

- [Consistency Models as a Rich and Efficient Policy Class for Reinforcement Learning](https://arxiv.org/abs/2309.16984)
  - Zihan Ding, Chi Jin
  - Key: Consistency policy, three typical RL settings
  - ExpEnv: D4RL, Gym MuJoCo

- [MADiff: Offline Multi-agent Learning with Diffusion Models](https://arxiv.org/abs/2305.17330)
  - Zhengbang Zhu, Minghuan Liu, Liyuan Mao, Bingyi Kang, Minkai Xu, Yong Yu, Stefano Ermon, Weinan Zhang
  - Key: Multi-agent, Offline RL, Classifier-free
  - ExpEnv: MPE, SMAC, Multi-Agent Trajectory Prediction (MATP)

- [Diffusion Policies for Out-of-Distribution Generalization in Offline Reinforcement Learning](https://arxiv.org/abs/2307.04726)
  - Suzan Ece Ada, Erhan Oztop, Emre Ugur
  - Key: Offline RL, OOD Generalization
  - ExpEnv: 2D-Multimodal Contextual Bandit, D4RL

- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
  - Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, Shuran Song
  - Key: Robot Manipulation
  - ExpEnv: Robomimic, Push-T, Multimodal Block Pushing, Franka Kitchen

- [Diffusion-based Generation, Optimization, and Planning in 3D Scenes](https://arxiv.org/abs/2301.06015)
  - Siyuan Huang, Zan Wang, Puhao Li, Baoxiong Jia, Tengyu Liu, Yixin Zhu, Wei Liang, Song-Chun Zhu
  - Key: 3D Scene Understanding, Optimization, Planning
  - Code: [official](https://github.com/scenediffuser/Scene-Diffuser)
  - ExpEnv: [ScanNet](http://www.scan-net.org/), [MultiDex](https://github.com/tengyu-liu/GenDexGrasp), [PROX](https://prox.is.tue.mpg.de/index.html)

- [Goal-Conditioned Imitation Learning using Score-based Diffusion Policies](https://arxiv.org/abs/2302.01877)
  - Zhixuan Liang, Yao Mu, Mingyu Ding, Fei Ni, Masayoshi Tomizuka, Ping Luo
  - Key: Goal-Conditioned Imitation Learning, Robotics, Classifier-free
  - ExpEnv: CALVIN, Block-Push, Relay Kitchen

### ICML 2024
- [Resisting Stochastic Risks in Diffusion Planners with the Trajectory Aggregation Tree](https://proceedings.mlr.press/v235/feng24b.html)
  - Lang Feng, Pengjie Gu, Bo An, Gang Pan
  - Publisher: ICML 2024
  - Key: diffusion planners, stochastic risk, tree, training-free
  - Code: [official](https://github.com/langfengQ/tree-diffusion-planner)
  - ExpEnv: Maze2D, MuJoco, D4RL

- [DiffStitch: Boosting Offline Reinforcement Learning with Diffusion-based Trajectory Stitching](https://proceedings.mlr.press/v235/liu24ao.html)
  - Guanghe Li, Yixiang Shan, Zhengbang Zhu, Ting Long, Weinan Zhang
  - Publisher: ICML 2024
  - Key: data augmentation pipeline, offline RL
  - Code: [official](https://github.com/guangheli12/DiffStitch)
  - ExpEnv: D4RL

- [Energy-Guided Diffusion Sampling for Offline-to-Online Reinforcement Learning](https://proceedings.mlr.press/v235/liu24ao.html)
  - Xu-Hui Liu, Tian-Shuo Liu, Shengyi Jiang, Ruifeng Chen, Zhilong Zhang, Xinwei Chen, Yang Yu
  - Publisher: ICML 2024
  - Key: data distribution shift, plug-in approach
  - Code: [official](https://github.com/liuxhym/EDIS)
  - ExpEnv: D4RL, Mujoco, AntMaze Navigation, Adroit Manipulation


### CVPR 2024

- [NIFTY: Neural Object Interaction Fields for Guided Human Motion Synthesis](https://openaccess.thecvf.com/content/CVPR2024/html/Kulkarni_NIFTY_Neural_Object_Interaction_Fields_for_Guided_Human_Motion_Synthesis_CVPR_2024_paper.html)
  - Nilesh Kulkarni, Davis Rempe, Kyle Genova, Abhijit Kundu, Justin Johnson, David Fouhey, Leonidas Guibas
  - Publisher: CVPR 2024
  - Key: 3D Motion Generation, Neural Interaction Fields, Human-Object Interaction
  - Code: [official](https://nileshkulkarni.github.io/nifty)
  - ExpEnv: AMASS

- [Hierarchical Diffusion Policy for Kinematics-Aware Multi-Task Robotic Manipulation](https://arxiv.org/abs/2403.03890)
  - Xiao Ma, Sumit Patidar, Iain Haughton, Stephen James
  - Publisher: CVPR 2024
  - Key: long-horizon task planning, diffusion models
  - Code: [official](https://github.com/dyson-ai/hdp)
  - ExpEnv: RLBench

### ICLR 2024

- [Training Diffusion Models with Reinforcement Learning](https://arxiv.org/pdf/2305.13301.pdf)
  - Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, Sergey Levine
  - Publisher: ICLR 2024
  - Key: reinforcement learning, RLHF, diffusion models
  - Code: [official](http://rl-diffusion.github.io/)
  - ExpEnv: None

- [Reasoning with Latent Diffusion in Offline Reinforcement Learning](https://arxiv.org/pdf/2309.06599.pdf)
  - Siddarth Venkatraman, Shivesh Khaitan, Ravi Tej Akella, John Dolan, Jeff Schneider, Glen Berseth
  - Publisher: ICLR 2024
  - Key: Reinforcement Learning, Diffusion Models
  - Code: [official](https://github.com/ldcq/ldcq)
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

- [DMBP: Diffusion model based predictor for robust offline reinforcement learning against state observation perturbations](https://openreview.net/forum?id=ZULjcYLWKe)
  - Anonymous Authors
  - Publisher:  ICLR 2024
  - Key: Robust Reinforcement Learning, Offline Reinforcement Learning, Diffusion Models
  - Code: [official](https://github.com/wrk8/DMBP)
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

- [Flow to Better: Offline Preference-based Reinforcement Learning via Preferred Trajectory Generation](https://openreview.net/forum?id=EG68RSznLT)
  - Zhilong Zhang, Yihao Sun , Junyin Ye, Tianshuo Liu, Jiaji Zhang, Yang Yu
  - Publisher:  ICLR 2024
  - Key: Preference-based Reinforcement Learning, Offline Reinforcement Learning, Conditional Generative Modeling, Diffusion Models
  - Code: [official]()
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)，[MetaWorld](https://github.com/Farama-Foundation/Metaworld)

- [Score Regularized Policy Optimization through Diffusion Behavior](https://arxiv.org/pdf/2310.07297.pdf)
  - Huayu Chen, Cheng Lu, Zhengyi Wang, Hang Su, Jun Zhu
  - Publisher:  ICLR 2024
  - Key: offline reinforcement learning, generative models, diffusion models, behavior modeling, computational efficiency
  - Code: [official](https://github.com/thu-ml/SRPO)
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

- [Simple Hierarchical Planning with Diffusion](https://arxiv.org/pdf/2401.02644.pdf)
  - Chang Chen, Fei Deng, Kenji Kawaguchi, Caglar Gulcehre, Sungjin Ahn
  - Publisher:  ICLR 2024
  - Key: Hierarchical Offline RL, Hierarchical planning, Hierarchical Reinforcement Learning, Diffusion-Based Planning
  - Code: [official](https://github.com/sail-sg/edp)
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

- [Efficient Planning with Latent Diffusion](https://openreview.net/forum?id=btpgDo4u4j)
  - Wenhao Li
  - Publisher:  ICLR 2024
  - Key: Offline Decision-Making, Offline Reinforcement Learning, Generative Model, Diffusion Model
  - Code: [official]()
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

- [Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion](https://arxiv.org/abs/2311.01017)
  - Lunjun Zhang, Yuwen Xiong, Ze Yang, Sergio Casas, Rui Hu, Raquel Urtasun
  - Publisher:  ICLR 2024
  - Key: discrete diffusion; world model; autonomous driving
  - Code: [official]()
  - ExpEnv: [NuScenes](https://www.nuscenes.org/), [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php), [Argoverse2 Lidar](https://www.argoverse.org/av2.html)

- [AlignDiff: Aligning Diverse Human Preferences via Behavior-Customisable Diffusion Model](https://openreview.net/forum?id=bxfKIYfHyx)
  - Zibin Dong, Yifu Yuan, Jianye Hao, Fei Ni, Yao Mu, Yan Zheng,Yujing Hu, Tangjie Lv, Changjie Fan, Zhipeng Hu
  - Publisher:  ICLR 2024
  - Key: Reinforcement learning; Diffusion models; RLHF; Preference aligning
  - Code: [official](https://aligndiff.github.io/)
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

### NeurIPS 2023

- [Learning Universal Policies via Text-Guided Video Generation](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1d5b9233ad716a43be5c0d3023cb82d0-Abstract-Conference.html)
  - Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, Pieter Abbeel
  - Publisher:  NeurIPS 2023
  - Key: Text-Guided Image Synthesis, Sequential Decision Making, Video Generation
  - ExpEnv: [real-world robotic]()

- [Diffusion Model is an Effective Planner and Data Synthesizer for Multi-Task Reinforcement Learning](https://openreview.net/pdf?id=fAdMly4ki5)
  - Haoran He, Chenjia Bai, Kang Xu, Zhuoran Yang, Weinan Zhang, Dong Wang, Bin Zhao, Xuelong Li
  - Publisher:  NeurIPS 2023
  - Key: multi-task reinforcement learning, diffusion models, planning, data synthesis
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

- [Learning Score-based Grasping Primitive for Human-assisting Dexterous Grasping](https://arxiv.org/pdf/2309.06038.pdf)
  - Tianhao Wu, Mingdong Wu, Jiyao Zhang, Yunchong Gan, Hao Dong
  - Publisher:  NeurIPS 2023
  - Key: Residual Policy Learning, Dexterous Grasping, Score-based Diffusion
  - Code: [official](https://github.com/tianhaowuhz/human-assisting-dex-grasp)
  - ExpEnv: [IsaacGym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/tree/main)
    
- [Efficient Diffusion Policies for Offline Reinforcement Learning](https://arxiv.org/abs/2305.20081)
  - Bingyi Kang, Xiao Ma, Chao Du, Tianyu Pang, Shuicheng Yan
  - Publisher:  NeurIPS 2023
  - Key: Computational Efficiency, Offline RL
  - Code: [official](https://github.com/sail-sg/edp)
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

### ICML 2023

- [Optimizing DDPM Sampling with Shortcut Fine-Tuning](https://arxiv.org/abs/2301.13362)
  - Ying Fan, Kangwook Lee
  - Publisher: ICML 2023
  - Key: Training Diffusion with RL, Online RL, Sampling Optimization
  - Code: [official](https://github.com/UW-Madison-Lee-Lab/SFT-PG)
  - ExpEnv: CIFAR10, CelebA

- [MetaDiffuser: Diffusion Model as Conditional Planner for Offline Meta-RL](https://arxiv.org/abs/2305.19923)
  - Fei Ni, Jianye Hao, Yao Mu, Yifu Yuan, Yan Zheng, Bin Wang, Zhixuan Liang
  - Publisher: ICML 2023
  - Key: Offline meta-RL, Conditional Trajectory Generation, Generalization, Classifier-guided
  - ExpEnv: [MuJoCo](https://github.com/openai/mujoco-py)

- [Hierarchical diffusion for offline decision making](https://proceedings.mlr.press/v202/li23ad.html)
  - Wenhao Li, Xiangfeng Wang, Bo Jin, Hongyuan Zha
  - Publisher: ICML 2023
  - Key: Hierarchical Offline RL, Long Horizon Task, Classifier-free
  - Code: [official](https://github.com/ewanlee/HDMI)
  - ExpEnv: [MuJoCo](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl), [NeoRL](https://github.com/polixir/NeoRL)

- [Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning](https://arxiv.org/abs/2304.12824)
  - Cheng Lu, Huayu Chen, Jianfei Chen, Hang Su, Chongxuan Li, Jun Zhu
  - Publisher: ICML 2023
  - Key: Offline RL, Constrained Policy Optimization, Classifier-guided
  - Code: [official](https://github.com/ChenDRAG/CEP-energy-guided-diffusion)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl)

### ICLR 2023

- [Is Conditional Generative Modeling all you need for Decision-Making?](https://arxiv.org/abs/2211.15657)
  - Anurag Ajay, Yilun Du, Abhi Gupta, Joshua Tenenbaum, Tommi Jaakkola, Pulkit Agrawal
  - Publisher: ICLR 2023
  - Key: Offline RL, Generative Model, Policy Optimization, Classifier-free
  - Code: [official](https://github.com/anuragajay/decision-diffuser/tree/main/code)
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

- [Imitating Human Behaviour with Diffusion Models](https://arxiv.org/abs/2301.10677)
  - Tim Pearce, Tabish Rashid, Anssi Kanervisto, Dave Bignell, Mingfei Sun, Raluca Georgescu, Sergio Valcarcel Macua, Shan Zheng Tan, Ida Momennejad, Katja Hofmann, Sam Devlin
  - Publisher: ICLR 2023
  - Key: Offline RL, Policy Optimization, Imitation Learning, Classifier-free
  - ExpEnv: Claw, Kitchen, CSGO

- [Offline Reinforcement Learning via High-Fidelity Generative Behavior Modeling](https://arxiv.org/abs/2209.14548)
  - Huayu Chen, Cheng Lu, Chengyang Ying, Hang Su, Jun Zhu
  - Publisher: ICLR 2023
  - Key: Offline RL, Generative models
  - Code: [official](https://github.com/ChenDRAG/SfBC)
  - ExpEnv: [D4RL](https://github.com/rail-berkeley/d4rl)

### ICRA 2023

- [Guided Conditional Diffusion for Controllable Traffic Simulation](https://arxiv.org/abs/2210.17366)
  - Ziyuan Zhong, Davis Rempe, Danfei Xu, Yuxiao Chen, Sushant Veer, Tong Che, Baishakhi Ray, Marco Pavone
  - Publisher: ICRA 2023
  - Key: Traffic Simulation, Multi-Agent, Classifier-free
  - ExpEnv: [nuScenes](https://github.com/nutonomy/nuscenes-devkit)

### NeurIPS 2022

- [TarGF: Learning Target Gradient Field to Rearrange Objects without Explicit Goal Specification](https://arxiv.org/abs/2209.00853)
  - Mingdong Wu, Fangwei Zhong, Yulong Xia, Hao Dong
  - Publisher:  NeurIPS 2022
  - Key: Inverse RL, Goal Specification, Score-based Diffusion
  - Code: [official](https://github.com/AaronAnima/TarGF)
  - ExpEnv: [Room Rearrangement](https://github.com/AaronAnima/TarGF/tree/main/envs/Room), [Ball Rearrangement](https://github.com/AaronAnima/EbOR)

- [Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning](https://arxiv.org/abs/2208.06193)
  - Zhendong Wang, Jonathan J Hunt, Mingyuan Zhou
  - Publisher:  NeurIPS Deep RL Workshop 2022
  - Key: Offline RL, Policy Optimization
  - Code: [official](https://github.com/zhendong-wang/diffusion-policies-for-offline-rl), [unofficial](https://github.com/twitter/diffusion-rl)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl)

### ICML 2022

- [Planning with Diffusion for Flexible Behavior Synthesis](https://arxiv.org/abs/2205.09991)
  - Michael Janner, Yilun Du, Joshua B. Tenenbaum, Sergey Levine
  - Publisher:  ICML 2022 (long talk)
  - Key: Offline RL,  Model-based RL, Trajectory Optimization, Classifier-guided
  - Code: [official](https://github.com/jannerm/diffuser)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl)

## Contributing

Our purpose is to make this repo even better. If you are interested in contributing, please refer to [HERE](CONTRIBUTING.md) for instructions in contribution.

## License

Awesome Diffusion Model in RL is released under the Apache 2.0 license.
