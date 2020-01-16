PPOConfig = dict(
    clip_param=0.5,
    ppo_epoch=4,  # TODO to be moved
    num_mini_batch=32,  # TODO to be moved
    value_loss_coef=0.5,
    entropy_coef=0.01,
)
