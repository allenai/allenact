
<div align="center">
    <img src="img/AllenAct.png" />
    <br>
    <i><h3>An open source framework for research in Embodied-AI</h3></i>
    </p>
    <hr/>
</div>

**AllenAct** is a modular and flexible learning framework designed with a focus on the unique requirements of Embodied-AI research. It provides first-class support for a growing collection of embodied environments, tasks and algorithms, provides reproductions of state-of-the-art models and includes extensive documentation, tutorials, start-up code, and pre-trained models.

AllenAct is built and backed by the [Allen Institute for AI (AI2)](https://allenai.org/). AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.

## Quick Links

- [Website & Docs](https://allenact.org/)
- [Github](https://github.com/allenai/allenact)
- [Install](https://allenact.org/installation/installation-allenact/)
- [Tutorials](https://allenact.org/tutorials/)
- [Citation](#citation)

## Features & Highlights

* _Support for multiple environments_: Support for the [i-THOR](https://ai2thor.allenai.org/ithor/), [Robo-THOR](https://ai2thor.allenai.org/robothor/) and [Habitat](https://aihabitat.org/) embodied environments as well as for grid-worlds including [MiniGrid](https://github.com/maximecb/gym-minigrid).
* _Task Abstraction_: Tasks and environments are decoupled in AllenAct, enabling researchers to easily implement a large variety of tasks in the same environment.
* _Algorithms_: Support for a variety of on-policy algorithms including [PPO](https://arxiv.org/pdf/1707.06347.pdf), [DD-PPO](https://arxiv.org/pdf/1911.00357.pdf), [A2C](https://arxiv.org/pdf/1611.05763.pdf), Imitation Learning and [DAgger](https://www.ri.cmu.edu/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf) as well as offline training such as offline IL.
* _Sequential Algorithms_: Trivial to experiment with different sequences of training routines, which are often the key to successful policies.
* _Simultaneous Losses_: Easily combine various losses while training models (e.g. use an external self-supervised loss while optimizing a PPO loss).
* _Multi-agent support_: Support for multi-agent algorithms and tasks.
* _Visualizations_: Out of the box support to easily visualize first and third person views for agents as well as intermediate model tensors, integrated into Tensorboard.
* _Pre-trained models_: Code and models for a number of standard Embodied AI tasks.
* _Tutorials_: Start-up code and extensive tutorials to help ramp up to Embodied AI.
* _First-class PyTorch support_: One of the few RL frameworks to target PyTorch.


## Contributions
We welcome contributions from the greater community. If you would like to make such a contributions we recommend first submitting an [issue](https://github.com/allenai/allenact/issues) describing your proposed improvement. Doing so can ensure we can validate your suggestions before you spend a great deal of time upon them. Improvements and bug fixes should be made via a pull request from your fork of the repository at [https://github.com/allenai/allenact](https://github.com/allenai/allenact).

All code in this repository is subject to formatting, documentation, and type-annotation guidelines. For more details, please see the our [contribution guidelines](CONTRIBUTING.md).

## Acknowledgments
This work builds upon the [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) library of Ilya Kostrikov and uses some data structures from FAIR's [habitat-api](https://github.com/facebookresearch/habitat-api).

## License
AllenAct is MIT licensed, as found in the [LICENSE](LICENSE.md) file.

## Team
AllenAct is an open-source project built by members of the PRIOR research group at the Allen Institute for Artificial Intelligence (AI2). 

<div align="left">
    <br>
    <img src="img/PRIORLogoBlackEmbedded.png">
     &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
    <img src="img/AI2_Logo_Square_Gradients_crop.png">
    <br>
</div>

## Citation
If you use this work, please cite:

```bibtex
@article{AllenAct,
  author = {Luca Weihs and Jordi Salvador and Klemen Kotar and Unnat Jain and Kuo-Hao Zeng and Roozbeh Mottaghi and Aniruddha Kembhavi},
  title = {AllenAct: A Framework for Embodied AI Research},
  year = {2020},
  journal = {arXiv},
}

```


