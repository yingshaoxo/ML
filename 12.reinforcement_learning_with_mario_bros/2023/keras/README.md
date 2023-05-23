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

> In mario, the big direction means a model that uses 50_steps_later_reward as guide. the current direction means another model that uses 4_steps_later_reward as guide. If two guide matchs, then do it!


## Summary
Short quick reaction for long term future


## Tips
* time_sequence_state_list(a series of images) is better than single frame
* use mongodb to cache data for supervised learning later to see if the model is able to fetch key features or not
* buy >= 8GB GPU


## In reality
### How to increase success rate?
Based on reward model output value, decrease optional actions by deleting those actions that most time causes negative values.


## New Theory
Instead of let the mario predict the reward of the future, how about we let them calculate current reward for each frame, that's a certain thing.

Then we use image to image state of art model to predict a future 50 steps later image. 

Maybe that will increase the success rate for mario.

## New Theory2
Based on New Theory 1, if we continue use image to image model to predict next frame, we'll be able to predict the future in a precise way.

```python
def predict_next_state(current_state, action_to_take):
    # image to image model
    # last layer multiply by action to take
    return next_state

def get_reward_based_on_this_state(current_state):
    # image to category number
    # for example, {0: bad, 1: normal, 2: good}
    return reward_category_number

def check_if_this_action_is_good_for_the_future_or_not(current_state, action_to_take, the_future_length=1):
    # it returns the scores for this action after the_future_length steps, it is a number between [-1, 1], if it == 1, it is good, if it == -1, it is bad
    reward_sum = 0

    next_state = current_state
    for i in range(the_future_length):
        next_state = predict_next_state(next_state, action_to_take)
        next_step_reward = get_reward_based_on_this_state(next_state)
        if next_step_reward == "good":
            reward_sum += 1
        elif next_step_reward == "normal":
            reward_sum += 0
        elif next_step_reward == "bad":
            reward_sum += -1
    
    return reward_sum / the_future_length

def get_a_series_of_actions_that_is_best_for_the_future(current_state, optional_actions_list, the_future_length=2) -> list[int]:
    """
    By using this function, you'll get an action list, by perform those different actions in the future, you'll get the maximum reward. (连招模式)
    """
    """
    This function will simply predict the next state, then call check_if_this_action_is_good_for_the_future_or_not() function to see if it is good or not. if it is good, based on that new_state, we call  check_if_this_action_is_good_for_the_future_or_not() function again to get the second best action, then we do it over and over again to get a series of different actions as a list for the best future.
    """
    pass
```

> **the `current_state` is `a_series_states_from_past_to_now`**

## New Theory3
How about we forget the past 1,2 theory. Instead, we use old method. We predict 200 steps later reward, but we don't wait one action for 200 steps.

For each state, we predict 200 steps later reward and react immediately according to those predictions.

> This is only for simple games. For complex games, we have to use 'deep decision tree' to get a series of actions to handle complex future.

___

## Super Thinking
What makes super human in intellegence?

1. Super observation. (Love to explore, love to watch and feel. See what others can't see easily. See core features or rules by simplification.)
2. Environment that has rewards and punishment. (Where do the right thing gets right reward)
3. Environment that allows you to take action and get result.
4. Big Storage.
5. High Computation Rate.

___

## How to make useable reinforment learning program?
1. Find useable model architecture by using supervised learning tech. (Make sure model can grab core features from data)
2. Find right positive and negative reward that could guide the agent to move forward.
3. Combine random_observation and perform_predicted_action and analyze_result and model_trainning into one loop.