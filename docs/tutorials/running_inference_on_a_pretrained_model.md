# Tutorial: Inference with a pre-trained model

In this tutorial we will run inference on a pre-trained model for the PointNav task
in the RoboTHOR environment. In this task the agent is tasked with going to a specific location
within a realistic 3D environment.

For information on how to train a PointNav Model see [this tutorial](training-a-pointnav-model.md)

We will need to [install the RoboTHOR environment](../installation/installation-allenact.md) and [download the 
RoboTHOR Pointnav dataset](../installation/download-datasets.md) before we get started.

For this tutorial we will download the weights of a pre trained model.
This can be done with a handy script in the `pretrained_model_ckpts` directory:
```bash
sh pretrained_model_ckpts/download_navigation_model_ckpts.sh robothor-pointnav-rgb-resnet
```
This will download the weights for an RGB model that has been
trained on the PointNav task in RoboTHOR to `pretrained_model_ckpts/robothor-pointnav-rgb-resnet.pt`


Next we need to run the inference, using the PointNav experiment config from the [tutorial on making a PointNav experiment](training-a-pointnav-model.md).
We can do this with the following command:

```bash
python main.py -o <PATH_TO_OUTPUT> -c <PATH_TO_CHECKPOINT> -b <BASE_DIRECTORY_OF_YOUR_EXPERIMENT> <EXPERIMENT_NAME>
```

Where `<PATH_TO_OUTPUT>` is the location where the results of the test will be dumped, `<PATH_TO_CHECKPOINT>` is the 
location of the downloaded model weights, `<BASE_DIRECTORY_OF_YOUR_EXPERIMENT>` is a path to the directory where 
our experiment definition is stored and `<EXPERIMENT_NAME>` is simply the name of our experiment
(without the file extension).
 
 For our current setup the following command would work:
 
 ```bash
 python main.py -o storage/ -c pretrained_model_ckpts/robothor-pointnav-rgb-resnet.pt -b projects/tutorials pointnav_robothor_rgb_ddppo
```

If we want to run inference on a model we trained ourselves, we simply have to point `<PATH_TO_CHECKPOINT>`
to the checkpoint we want to evaluate.
