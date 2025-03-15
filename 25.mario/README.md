# Use hard coding method to play mario

## Install
```
pip install gym-super-mario-bros=7.4.0

pip install gym==0.17.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy==1.26 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pillow -i https://pypi.tuna.tsinghua.edu.cn/simple

conda install -c conda-forge gcc
```

yingshaoxo: since the NES game is a relatively fixed env game type. So even if you use some kind of picture match algorithm to replay a list of action, it is also possible for mario to win the game. You just have to "teach" the mario right things. (picture match: it is like asking you to find a dog in a white paper, you just have to let the computer remember a small picture of a dog, then ask the computer to look at the paper, from left to right, from top to bottom, then the problem will get solved.)