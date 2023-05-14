# We use keras to play mario this time

## Set up Env
### Linux
```
sudo apt install python3.10 -y
sudo apt install pip -y

python3.10 -m pip install -r requirements.txt
```

## Theory0
0. we use random action to start the game
1. we record (state, action, reward_after_perform_the_action)
2. we train the reward_model with the data we recorded
3. we choose the max_reward action by using the reward_model, meanwhile, we record (state, max_reward_action) to train actor_model, we still record (state, action, reward_after_perform_the_action)
4. we use actor_model to get recommended action, perform it, and we still record (state, action, reward_after_perform_the_action)

### Simpler Thoery
1. we use random action to train reward_model. (state, action -> reward in the future)
2. we use reward_model to get max_reward action to train actor_model, reward_model. (we do multiple prediction to find every action's future reward, then choose the action that leads to maximum reward in the future)
3. we use actor_model to get more action_and_reward_data to train reward_model. (for each observation and act, we'll get more experimence for the reward_model, what action cause what reward in the future, the reward_model will be more and more accurate)

> 3 functions in a loop


## Theory2
Based on theory1, we no longer predict exact reward value, but 3 classes: 'good to me', 'bad to me', 'no good and also no bad to me'


## Theory3
Three level 'optimal path' finding, use thoery2 as the direction, see what is right, what is wrong, in the far future. use theory0 to check it again, to see what we should do in the current moment

> Imagine you are doing coding for a remote control car, you use GPS to find the 'big direction' for the target, then you use camera to find the 'black line', then you use the 'white and black color senser (grayscale sensor)' to guide your car go that way in an exact way. Actually you are using three models togather at the same time to make sure the car will definitely go towards the target.

> In mario, the big direction means a model that use 50 steps later reward as guide. the current direction means another model that use 4 steps later reward as guide. Two guide match, then do it!


## Tips
* time_sequence_state_list(a series of images) is better than single frame
* use mongodb to cache data for supervised learning later to see if the model is able to fetch key features or not
* buy >= 8GB GPU