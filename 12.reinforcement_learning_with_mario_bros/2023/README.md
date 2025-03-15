# Use `Pytorch` and `Proximal Policy Optimization` to play mario

## Set up Env
```bash
sudo apt install python3.10 -y
sudo apt install pip -y

python3.10 -m pip install -r requirements.txt
```

## Install
```
pip install gym-super-mario-bros=7.4.0

pip install gym==0.17.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy==1.26 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pillow -i https://pypi.tuna.tsinghua.edu.cn/simple

conda install -c conda-forge gcc
```

## Run it

I use hard coding method to play mario.

For `yingshaoxo_method.py`

Press number 6 to let AI to play mario

Press number 0, 1, 2 to manually play and add data for mario

Press q to quit


## Advanced Solution 1

Train 2 models, one main-AI is used to pass the game, another warning-AI is used to do the warning (so the main-AI would know whether to play it randomly or predictively).

When the main-AI got trouble, pause in there. main-AI forget that short-period-of-memory. But the warning-AI have to rememeber that short-period-of-memory with the target-output setted to 'failure', so it can send the warning next time.

~~main-AI redo/retry to pass that trouble randomly, until success. if success, main-AI remember it for n times.~~

When the warning-AI send the warning, the main-AI start to play randomly until success. If success, send the data to the warning-ai with the target-output setted to 'success'.

> For game company, When the warning-AI send the warning, without going back to the original position, from the pause position, the main-AI play it normally for one time, if not success, play it randomly again. 

## Advanced Solution 2

Train sub-module, for example, if you are an AI, for each city, you its corresponding module to adopt it, so you can be a master of that city, and you'll never fail.

The same logic applies to NES games, for each level, you train different models.
