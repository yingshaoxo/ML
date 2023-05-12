# We use keras to play mario this time

## Set up Env
### Linux
```
sudo apt install python3.10 -y
sudo apt install pip -y

python3.10 -m pip install -r requirements.txt
```

## Theory
1. actor_model takes (state), returns (action_probability_list); reward_model takes (state, action), returns (reward)
<!-- 2. why reward_model could make the actor_model perform better over time??? -->
2. we train the actor_model by giving it x=(state), y=(the optimal action predicted by reward_model)