# Use `Pytorch` and `Proximal Policy Optimization` to play mario

## Set up Env
```bash
sudo apt install python3.10 -y
sudo apt install pip -y

python3.10 -m pip install -r requirements.txt
```

## Run it
```bash
python3.10 main.py
```

## Advanced Solution 1

Train 2 models, one main-AI is used to pass the game, another warning-AI is used to do the warning (so the main-AI would know whether to play it randomly or predictively).

When the main-AI got trouble, pause in there. main-AI forget that short-period-of-memory. But the warning-AI have to rememeber that short-period-of-memory with the target-output setted to 'failure', so it can send the warning next time.

~~main-AI redo/retry to pass that trouble randomly, until success. if success, main-AI remember it for n times.~~

When the warning-AI send the warning, the main-AI start to play randomly until success. If success, send the data to the warning-ai with the target-output setted to 'success'.

> For game company, When the warning-AI send the warning, without going back to the original position, from the pause position, the main-AI play it normally for one time, if not success, play it randomly again. 

## Advanced Solution 2

Train sub-module, for example, if you are an AI, for each city, you its corresponding module to adopt it, so you can be a master of that city, and you'll never fail.

The same logic applies to NES games, for each level, you train different models.