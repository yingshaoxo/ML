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

## Usage
For `1.3.cpu_quick_mode.py`

Press number 6 to let AI to play mario

Press number 0, 1, 2 to manually play and add data for mario

## Limitation
The current image similarity compare method is not efficient, and has low accuracy.

If you can improve it to human_level, then train a mario AI agent will only require 1 minute.

## Author
yingshaoxo

## Final Thinking

```
If you want to let the mario train itself,

You have to use random actions after it reachs `last 1/10 farest_distance`.

Then do the data saving for `[farest_distance, farest_distance + (farest_distance // 2)]`.

If it reachs the `(farest_distance * 2)`, the `new farest_distance` would be equal to `farest_distance + (farest_distance // 2)`.

> You can actually pre_generate random action sequence and test it without gpu, then save it to database if it is working. It would be 100x faster than using a GPU.
```

```
New mario learning method:


Split the screen to 16 square images, take a bunch of screenshot.

Use unsupervised learning (k-clustering) to get 512 different classes.

So that for each timestep, you can always get 16 length number list. (Each element is a number between 0-511)

Then according to human_play data, you replay those actions according to the 16 length number list state.


#ai #dev #programming #yingshaoxo
```
