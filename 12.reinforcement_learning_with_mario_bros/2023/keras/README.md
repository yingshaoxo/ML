# We use keras to play mario this time

## Set up Env
### Linux
```
sudo apt install python3.10 -y
sudo apt install pip -y

python3.10 -m pip install -r requirements.txt
```

## Theory
0. we use random action to start the game
1. we record (state, action, reward_after_perform_the_action)
2. we train the reward_model with the data we recorded
3. we choose the max_reward action by using the reward_model, meanwhile, we record (state, max_reward_action) to train actor_model, we still record (state, action, reward_after_perform_the_action)
4. we use actor_model to get recommended action, we still record (state, action, reward_after_perform_the_action)

### Simpler Thoery
1. we use random action to train reward_model
2. we use reward_model to get max_reward action to train actor_model, reward_model
3. we use actor_model to get action to train reward_model

> 3 functions in a loop


## Theory2
Based on theory1, we no longer predict exact reward value, but 3 classes: 'good to me', 'bad to me', 'no good and also no bad to me'