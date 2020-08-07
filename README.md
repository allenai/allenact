# Embodied-AI
`embodied-ai` is a library designed for research in embodied AI with a focus on modularity and flexibility. 

### Features & Highlights

* _Decoupled tasks and environments_: In embodied AI research it is important to be able to define many tasks for a single environment; for instance, the [AI2-THOR](https://ai2thor.allenai.org/) environment has been used with tasks such as  
    * [semantic/object navigation](https://arxiv.org/abs/1810.06543),
    * [interactive question answering](https://arxiv.org/abs/1712.03316),
    * [multi-agent furniture lifting](https://prior.allenai.org/projects/two-body-problem), and
    * [adversarial hide-and-seek](https://arxiv.org/abs/1912.08195). 

    We have designed `embodied-ai` to easily support a wide variety of tasks designed for individual environments.

* _Support for several environments_: We support different environments used for Embodied AI research such as [AI2-THOR](https://ai2thor.allenai.org/), [Habitat](https://aihabitat.org/) and [MiniGrid](https://github.com/maximecb/gym-minigrid). We have made it easy to incorporate new environments.
* _Different input modalities_: The framework supports a variety of input modalities such as RGB images, depth, language and GPS readings. 
* _Various training pipelines_: The framework includes not only various training algorithms (A2C, PPO, DAgger, etc.) but also a mechanism to integrate different types of algorithms (e.g., imitation learning followed by reinforcement learning). 
* _First-class PyTorch support_: While many well-developed libraries exist for reinforcement learning in 
   Tensorflow, we are one of a few to target PyTorch.

### Support

`embodied-ai` currently supports the following environments, tasks, and algorithms.  We are actively working on integrating recently developed models and frameworks. Nevertheless, we provide tutorials to make it straightforward to integrate the algorithms, tasks, or environments of your choice. 

  |   Environments             |      Tasks      |   Algorithms    |
  | -------------------------- | --------------- | --------------- |
  | [iTHOR](https://ai2thor.allenai.org/ithor/), [RoboTHOR](https://ai2thor.allenai.org/robothor/), [Habitat](https://aihabitat.org/), [MiniGrid](https://github.com/maximecb/gym-minigrid) | [PointNav](https://arxiv.org/pdf/1807.06757.pdf), [ObjectNav](https://arxiv.org/pdf/2006.13171.pdf), [MiniGrid tasks](https://github.com/maximecb/gym-minigrid), [ALFRED](https://arxiv.org/pdf/1912.01734.pdf)  | [A2C](https://arxiv.org/pdf/1611.05763.pdf), [PPO](https://arxiv.org/pdf/1707.06347.pdf), [DD-PPO](https://arxiv.org/pdf/1911.00357.pdf), [DAgger](https://www.ri.cmu.edu/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf) |




### Quick links
* [Github Repository](https://github.com/allenai/embodied-rl)
* [Installation - TODO]()
* [Pre-trained Models - TODO]()

## Citation
If you use this work, please cite:

```text
@article{embodied-ai,
  author = {Luca Weihs and Jordi Salvador and Klemen Kotar and Unnat Jain and Kuo-Hao Zeng and Roozbeh Mottaghi and Aniruddha Kembhavi},
  title = {A Framework for Reproducible, Reusable, and Robust Embodied AI Research},
  year = {2020},
  journal = {arXiv},
}

```

<!-- ## Acknowledgments
This work builds upon the [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) library of Ilya Kostrikov and uses some data structures from FAIR's [habitat-api](https://github.com/facebookresearch/habitat-api).
 -->