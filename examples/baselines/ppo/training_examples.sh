# PokeCube-v1
for seed in 0 42 3407
do
    CUDA_VISIBLE_DEVICES=0 python ppo.py --env_id="PokeCube-v1" --num_envs=512 --update_epochs=8 --num_minibatches=32 --total_timesteps=10_000_000 --eval_freq=10 --num-steps=20 --wandb_entity="boyuanchen21-tsinghua-university" --track --seed=$seed
    CUDA_VISIBLE_DEVICES=0 python ppo_adapt.py --env_id="PokeCube-v1" --num_envs=512 --update_epochs=8 --num_minibatches=32 --total_timesteps=10_000_000 --eval_freq=10 --num-steps=20 --wandb_entity="boyuanchen21-tsinghua-university" --track --seed=$seed
    CUDA_VISIBLE_DEVICES=0 python ppo_rgb.py --env_id="PokeCube-v1" --num_envs=512 --update_epochs=8 --num_minibatches=32 --total_timesteps=20_000_000 --eval_freq=10 --num-steps=20 --wandb_entity="boyuanchen21-tsinghua-university" --track --seed=$seed
done