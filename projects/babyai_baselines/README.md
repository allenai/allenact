# Baseline experiments for the BabyAI environment

We perform a collection of baseline experiments run in the BabyAI environment
 on the GoToLocal task, see the `projects/babyai_baselines/experiments/go_to_local` directory.
 For instance, to train a model using PPO, please run
 
```bash
python ddmain.py go_to_local.ppo --experiment_base projects/babyai_baselines/experiments
```