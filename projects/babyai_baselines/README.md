# Baseline experiments for the BabyAI environment

We perform a collection of baseline experiments run in the BabyAI environment
 on the GoToLocal task, see the `projects/babyai_baselines/experiments/go_to_local` directory.
 For instance, to train a model using PPO, please run
 
```bash
python ddmain.py go_to_local.ppo --experiment_base projects/babyai_baselines/experiments
```

Note that these experiments will be quite slow when not using a GPU as the BabyAI model architecture is surprisingly 
large. Specifying a GPU (if available) can be done from the command line using hooks we created using 
[gin-config](https://github.com/google/gin-config). E.g. to train using the 0th GPU device, add

```bash
--gp "machine_params.gpu_id = 0"
```  

to the above command.