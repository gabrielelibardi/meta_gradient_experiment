
- Make compatible with use-proper-time-limits = False too.


python main/train_ppo_bullet.py --seed 18 --device 'cuda:0' --use-gae --lr 2e-4 --clip-param 0.2 --value-loss-coef 0.3 --num-processes 12 --num-steps 2048 --num-mini-batch 32 --entropy-coef 0.02 --num-env-steps 60000000 --log-dir ../RUNS/exp_test_ll --frame-stack 3  --gamma 0.99   --save-interval 50 --gae-lambda 0.95 --ppo-epoch 1