# I would recommand you to use tf.keras, because it is easyer to understand
import torch
import numpy

class MarioNeuralNetwork(torch.nn.Module):
    def __init__(self):
        from sys import platform
        import os
        super().__init__()

        if platform == "linux" or platform == "linux2":
            # linux
            self.device = "cuda:0"
        elif platform == "darwin":
            # OS X
            self.device = "mps"
        elif platform == "win32":
            # Windows...
            self.device = "cpu"
        
        print(self.device)
        self.device = torch.device(self.device)

        self.the_action_numbers = 7

        square_screen_one_side_length = 84

        kernel_size_of_first_convolutional_layer = int(square_screen_one_side_length / 4 / 2)

        image_input_channel_for_grayscale_image = 1
        category_number_for_the_first_layer_output = 7 * int(square_screen_one_side_length/kernel_size_of_first_convolutional_layer) * int(square_screen_one_side_length/kernel_size_of_first_convolutional_layer)
        # (7 categories) * (10 vertical positions) * (10 horizontal positions) 

        self.two_dimension_mario_position_for_one_single_image_linear_relu_stack = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=image_input_channel_for_grayscale_image, 
                out_channels=category_number_for_the_first_layer_output, 
                kernel_size=kernel_size_of_first_convolutional_layer
            ),
            torch.nn.ReLU(),
            # for grb image, the input channel is 3, for greyscale, the input channel is 1
            # output_channel here: how many final neural nodes you want to output, in other words, how many numbers you want to get from the input; or you can think it as how many categories you want to learn from the input
            torch.nn.Flatten(),
            torch.nn.Linear(5625, 540),
            torch.nn.ReLU(),
            torch.nn.Linear(540, 32),
            torch.nn.ReLU(),
        )

        # you'll need to add speed(positive means go right, negative means go left), and position to the next layer, so it can produce the final predicted action
        # for this game, you should also input the level number (1 to 8)
        # self.combined_features = torch.nn.Sequential(
        #     # change this input nodes
        #     torch.nn.Linear(3048, 512),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(512, 32),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(32,1))

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(14338, 100),
            torch.nn.Linear(100, self.the_action_numbers),
            torch.nn.LogSoftmax(dim=0)
        )

    def forward(self, picture, speed, game_level):
        graph_data = self.two_dimension_mario_position_for_one_single_image_linear_relu_stack(picture.to(self.device))
        graph_data = graph_data.reshape(-1)
        # graph_data = self.flatten_layer(graph_data) # not work compared to keras, maybe it's not flatten function

        speed = torch.tensor([speed]).to(self.device)
        game_level = torch.tensor([game_level]).to(self.device)

        merge_array = torch.cat((graph_data, speed, game_level), 0)
        # merge_array = self.combined_features(merge_array)

        data_for_output = self.output_layer(merge_array)
        return data_for_output
    

class Trainer():
    def __init__(self):
        from sys import platform
        import os

        self.mario_model = MarioNeuralNetwork()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.mario_model.parameters(), lr=0.001, momentum=0.9)

        if platform == "linux" or platform == "linux2":
            # linux
            self.device = "cuda:0"
        elif platform == "darwin":
            # OS X
            self.device = "mps"
        elif platform == "win32":
            # Windows...
            self.device = "cpu"
        
        print(self.device)
        self.device = torch.device(self.device)
        self.mario_model.to(self.device)

        self.epoch = 0
        self.need_to_update_lose = False
        self.loss_info = None

        self.action_identity = numpy.identity(
            self.mario_model.the_action_numbers
        )  # for quickly get a hot vector, like 0001000000000000

    def train(self, data, target_data):
        """
        data:  
            (
                graph_data: array[86,86,1], 
                speed: int, 
                game_level: int
            )
        target_data: 
            action: int
        """
        if (data[0] == None):
            return

        self.optimizer.zero_grad()
        output = self.mario_model(*data)
        # target_data = self.action_identity[target_data : target_data + 1]
        target_data = torch.tensor(target_data).to(self.device)
        loss = torch.nn.functional.nll_loss(output, target_data)
        loss.backward()
        self.optimizer.step()

        if self.need_to_update_lose == True:
            self.loss_info = loss.item()
            self.need_to_update_lose = False
            print('Train Epoch: {} \tLoss: {:.6f}'.format(self.epoch, loss.item()))
        
    def print_progress_info(self, epoch):
        self.epoch = epoch
        self.need_to_update_lose = True
    
    def save_model(self):
        torch.save(self.mario_model.state_dict(), "mario_model_made_by_yingshaoxo.pt") # pt == pytorch here

    def predict(self, data) -> int:
        """
        data:  
            (
                graph_data: array[86,86,1], 
                speed: int, 
                game_level: int
            )

        output: int
        """
        output = self.mario_model(*data)
        output = output.argmax()
        return int(output.to(torch.long))
    



# model = NeuralNetwork().to(device)
# print(model)

# X = torch.rand(1, 28, 28, device=device)
# logits = model(X)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")

# > Predicted class: tensor([9], device='cuda:0')

"""
def model_generate_function(num_inputs, num_actions):
    model = tf.keras.models.Sequential(
        [
            hub.KerasLayer(
                "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/feature_vector/2",
                trainable=False,
                input_shape=(num_inputs),
            ),
            # tf.keras.layers.Convolution2D(32, 3, 2, input_shape=(img_rows , img_cols), padding='same'),
            # tf.keras.layers.Activation('relu'),
            # tf.keras.layers.Convolution2D(32, 3, 2, padding='same'),
            # tf.keras.layers.Activation('relu'),
            # tf.keras.layers.Convolution2D(32, 3, 2, padding='same'),
            # tf.keras.layers.Activation('relu'),
            # tf.keras.layers.Convolution2D(32, 3, 2, padding="same"),
            # tf.keras.layers.Activation("relu"),
            # tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation=None),
            tf.keras.layers.Dense(num_actions, activation="softmax"),  # actor_linear
            # tf.keras.layers.Dense(1, activation=None), #critic_linear
        ]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    return model
"""


"""
class PPO(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PPO, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.linear(x.view(x.size(0), -1))
        return self.actor_linear(x), self.critic_linear(x)

"""
