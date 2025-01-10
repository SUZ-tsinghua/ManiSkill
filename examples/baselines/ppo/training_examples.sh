# # PokeCube-v1
# for seed in 42 #3407
# do
#     CUDA_VISIBLE_DEVICES=5 python ppo.py --env_id=PokeCube-v1 --num_envs=512 --update_epochs=8 --num_minibatches=32 --total_timesteps=14_000_000 --eval_freq=10 --num-steps=20 --wandb_entity=boyuanchen21-tsinghua-university --track --seed=${seed}
#     CUDA_VISIBLE_DEVICES=2 python ppo_adapt.py --env_id=PokeCube-v1 --num_envs=512 --update_epochs=8 --num_minibatches=32 --total_timesteps=14_000_000 --eval_freq=10 --num-steps=20 --wandb_entity=boyuanchen21-tsinghua-university --track --seed=${seed}
#     CUDA_VISIBLE_DEVICES=7 python ppo_rgb.py --env_id=PokeCube-v1 --num_envs=512 --update_epochs=8 --num_minibatches=32 --total_timesteps=30_000_000 --eval_freq=10 --num-steps=20 --wandb_entity=boyuanchen21-tsinghua-university --track --seed=${seed}
# done

# # RollBall-v1
# for seed in 3407
# do
#     CUDA_VISIBLE_DEVICES=7 python ppo.py --env_id=RollBall-v1 --num_envs=512 --update_epochs=8 --num_minibatches=32 --total_timesteps=30_000_000 --num-steps=80 --num_eval_steps=80 --gamma=0.95 --wandb_entity=boyuanchen21-tsinghua-university --track --seed=${seed}
#     CUDA_VISIBLE_DEVICES=4 python ppo_adapt.py --env_id=RollBall-v1 --num_envs=512 --update_epochs=8 --num_minibatches=32 --total_timesteps=30_000_000 --num-steps=80 --num_eval_steps=80 --gamma=0.95 --wandb_entity=boyuanchen21-tsinghua-university --track --seed=${seed}
#     CUDA_VISIBLE_DEVICES=6 python ppo_rgb.py --env_id=RollBall-v1 --num_envs=512 --update_epochs=8 --num_minibatches=32 --total_timesteps=30_000_000 --num-steps=80 --num_eval_steps=80 --gamma=0.95 --wandb_entity=boyuanchen21-tsinghua-university --track --seed=${seed}
# done

# # PickFromCabinetDrawer-v1
# # test version. Use a large total_timesteps
# for seed in 0 42 3407
# do 
#     CUDA_VISIBLE_DEVICES=0 python ppo.py --env_id=PickFromCabinetDrawer-v1 --num_envs=512 --update_epochs=8 --num_minibatches=32 --total_timesteps=100_000_000 --num-steps=100 --num_eval_steps=200 --gamma=0.95 --save-model --wandb_entity=boyuanchen21-tsinghua-university --track --seed=${seed}
# done

# # UnitreeG1TransportBox-v1
# for seed in 0
# do
#     # CUDA_VISIBLE_DEVICES=3 python ppo.py --env_id=UnitreeG1TransportBox-v1 --num_envs=512 --update_epochs=8 --num_minibatches=32 --total_timesteps=100_000_000 --num-steps=100 --num-eval-steps=200 --num_eval_envs=16 --gamma=0.95 --wandb_entity=boyuanchen21-tsinghua-university --track --seed=${seed}
#     CUDA_VISIBLE_DEVICES=5 python ppo_adapt.py --env_id=UnitreeG1TransportBox-v1 --num_envs=512 --update_epochs=8 --num_minibatches=32 --total_timesteps=100_000_000 --num-steps=100 --num-eval-steps=200 --num_eval_envs=16 --gamma=0.95 --wandb_entity=boyuanchen21-tsinghua-university --track --seed=${seed}
#     # CUDA_VISIBLE_DEVICES=4 python ppo_rgb.py --env_id=UnitreeG1TransportBox-v1 --num_envs=512 --update_epochs=8 --num_minibatches=32 --total_timesteps=100_000_000 --num-steps=100 --num-eval-steps=200 --num_eval_envs=16 --gamma=0.95 --wandb_entity=boyuanchen21-tsinghua-university --track --seed=${seed}
# done

# UnitreeG1TransportBox-v1
for seed in 0
do
    CUDA_VISIBLE_DEVICES=1 python ppo.py --env_id=StackCube-v1 --num_envs=512 --update_epochs=8 --num_minibatches=32 --total_timesteps=50_000_000 --num-eval-steps=100 --num_eval_envs=16 --wandb_entity=boyuanchen21-tsinghua-university --track --seed=${seed}
    # CUDA_VISIBLE_DEVICES=5 python ppo_adapt.py --env_id=StackCube-v1 --num_envs=512 --update_epochs=8 --num_minibatches=32 --total_timesteps=50_000_000 --num-eval-steps=100 --num_eval_envs=16 --wandb_entity=boyuanchen21-tsinghua-university --track --seed=${seed}
    # CUDA_VISIBLE_DEVICES=0 python ppo_rgb.py --env_id=StackCube-v1 --num_envs=512 --update_epochs=8 --num_minibatches=32 --total_timesteps=50_000_000 --num-eval-steps=100 --num_eval_envs=16 --wandb_entity=boyuanchen21-tsinghua-university --track --seed=${seed}
done
