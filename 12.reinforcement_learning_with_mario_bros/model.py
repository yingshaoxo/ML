import tensorflow as tf

img_rows , img_cols = 240, 256
number_of_actions = 7

def generate_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Convolution2D(32, 8, 8, input_shape=(img_rows, img_cols, 3)),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Convolution2D(64, 4, 4),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Convolution2D(64, 3, 3),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(number_of_actions, activation=tf.nn.softmax),
        #tf.keras.layers.Dense(1, activation="linear")
    ])

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])

    return model
