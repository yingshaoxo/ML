# How to play mario in m1 book at 2022

## env
```bash
conda env update --prefix ./env --file environment.yml  --prune

poetry install

python train.py --world 5 --stage 2 --lr 1e-4
```

## thanks to 
https://github.com/uvipen/Super-mario-bros-PPO-pytorch