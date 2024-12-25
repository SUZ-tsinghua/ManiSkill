# PickFromCabinetDrawer-v1
for seed in 0 #3407
do
    # CUDA_VISIBLE_DEVICES=4 python ppo.py --env_id="PickFromCabinetDrawer-v1"  --num_envs=512 --update_epochs=8 --num_minibatches=32 --total_timesteps=50_000_000 --num-steps=100 --num_eval_steps=200 --gamma=0.95 # --save-model --wandb_entity="boyuanchen21-tsinghua-university" --track --seed=${seed}
    CUDA_VISIBLE_DEVICES=49 python ppo_adapt.py --env_id="PickFromCabinetDrawer-v1"  --num_envs=512 --update_epochs=8 --num_minibatches=32 --total_timesteps=50_000_000 --num-steps=100 --num_eval_steps=200 --gamma=0.95 --save-model --wandb_entity="boyuanchen21-tsinghua-university" --track --seed=${seed}
    # CUDA_VISIBLE_DEVICES=5 python ppo_rgb.py --env_id="PickFromCabinetDrawer-v1"  --num_envs=512 --update_epochs=8 --num_minibatches=32 --total_timesteps=50_000_000 --num-steps=100 --num_eval_steps=200 --gamma=0.95  --save-model --wandb_entity="boyuanchen21-tsinghua-university" --track --seed=${seed}
done