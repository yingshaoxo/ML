import tensorflow as tf
import tensorflow_hub as hub


def generate_model(num_inputs, num_actions):
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
