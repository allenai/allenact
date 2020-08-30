# Baseline experiments for the BabyAI environment

We perform a collection of baseline experiments within the BabyAI environment
 on the GoToLocal task, see the `projects/babyai_baselines/experiments/go_to_local` directory. For the details of the task, refer to [this paper](https://openreview.net/pdf?id=rJeXCo0cYX).
 For instance, to train a model using PPO, run
 
```bash
python main.py go_to_local.ppo --experiment_base projects/babyai_baselines/experiments
```

Note that these experiments will be quite slow when not using a GPU as the BabyAI model architecture is 
large. Specifying a GPU (if available) can be done from the command line using hooks we created using 
[gin-config](https://github.com/google/gin-config). E.g. to train using the 0th GPU device, add

```bash
--gp "machine_params.gpu_id = 0"
```  

to the above command.
