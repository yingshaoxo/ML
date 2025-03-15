import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from auto_everything.image import Image
image = Image()
from auto_everything.disk import Disk
disk = Disk()

version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
  import tf_keras as keras
else:
  keras = tf.keras

keras.utils.disable_interactive_logging()

def get_mobilenet_model(model_path):
    mobilenet_model = keras.models.Sequential([
        hub.KerasLayer("./mobilenet_v3_small_tf2_feature_vector_224_224_to_1024", trainable=False),
    ])
    mobilenet_model.build([None, 224, 224, 3])
    return mobilenet_model
    """
    if not disk.exists(model_path):
        mobilenet_model = keras.models.Sequential([
            #mobilenet-v3-tensorflow2-large-100-224-feature-vector-v1
            #hub.KerasLayer("https://www.kaggle.com/models/google/mobilenet-v3/TensorFlow2/small-075-224-feature-vector/1", trainable=False),
            hub.KerasLayer("./mobilenet_v3_small_tf2_feature_vector_224_224_to_1024", trainable=False),
            #keras.layers.Dense(16, use_bias=False, activation='softmax')
        ])
        mobilenet_model.build([None, 224, 224, 3])
        mobilenet_model.save(model_path)
    else:
        mobilenet_model = keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    return mobilenet_model
    """

def the_tf_mobile_net_image_preprocess(data):
    # to 0 and 1, a float, it seems like
    data /= 127.5
    data -= 1.
    return data

def get_image_feature_vector_data(model, a_image):
    a_image = a_image.copy()
    a_image = a_image.resize(224, 224)

    img_data = np.asarray(a_image.raw_data, "f")
    img_data.flags.writeable = True
    img_data = img_data[:,:,0:3]
    img_data = np.expand_dims(img_data, axis=0)
    #img_data = keras.applications.mobilenet_v2.preprocess_input(img_data) # use v3 will cause bug
    img_data = the_tf_mobile_net_image_preprocess(img_data)

    mobilenet_feature = model.predict(img_data)
    mobilenet_feature_np = np.array(mobilenet_feature)
    final_data = mobilenet_feature_np.flatten()

    return final_data

def get_similarity_between_two_numpy_array(array_a, array_b):
    norm_a = np.linalg.norm(array_a)
    norm_b = np.linalg.norm(array_b)
    dot = sum(a * b for a, b in zip(array_a, array_b))
    return (dot / (norm_a * norm_b))

"""
The final goal of this script is to check if two image are the same, for example, in a photo, I think two inner image are the same, they are white, but computer or rgb color doesn't think so. We need to fix this problem.
"""

if __name__ == "__main__":
    mobile_net_model = get_mobilenet_model("./yingshaoxo_mobilenet_v3_small_tf2_feature_vector_224_224_to_1024")

    #source_image_path = "/home/yingshaoxo/Downloads/1.png"
    #source_image_path = "/home/yingshaoxo/Downloads/pillow.bmp"
    #source_image_path1 = "/home/yingshaoxo/Downloads/has_human.bmp"
    #source_image_path = "/home/yingshaoxo/Downloads/dnf_0.bmp"
    #source_image_path2 = "/home/yingshaoxo/Downloads/no_human.bmp"
    #source_image_path3 = "/home/yingshaoxo/Downloads/2286.bmp"
    #source_image_path3 = "/home/yingshaoxo/Downloads/food.bmp"
    #source_image_path3 = "/home/yingshaoxo/Downloads/tf_image_1.bmp"
    #source_image_path3 = "/home/yingshaoxo/Downloads/tf_image_2.bmp"
    #source_image_path = "/home/yingshaoxo/Downloads/hero2_simplified.png"
    #source_image_path = "/home/yingshaoxo/Downloads/a_girl.bmp"

    source_image_path1 = "/home/yingshaoxo/Downloads/tree1.bmp"
    source_image_path2 = "/home/yingshaoxo/Downloads/tree2.bmp"
    source_image_path3 = "/home/yingshaoxo/Downloads/clothe2.bmp"

    #source_image_path1 = "/home/yingshaoxo/Downloads/clothe1.bmp"
    #source_image_path2 = "/home/yingshaoxo/Downloads/clothe2.bmp"
    #source_image_path3 = "/home/yingshaoxo/Downloads/tree2.bmp"

    a_image1 = image.read_image_from_file(source_image_path1)
    a_image2 = image.read_image_from_file(source_image_path2)
    a_image3 = image.read_image_from_file(source_image_path3)

    data1 = get_image_feature_vector_data(mobile_net_model, a_image1)
    data2 = get_image_feature_vector_data(mobile_net_model, a_image2)
    data3 = get_image_feature_vector_data(mobile_net_model, a_image3)

    #print("very high", get_similarity_between_two_numpy_array(data1, data1))
    print("high", get_similarity_between_two_numpy_array(data1, data2))
    print("low", get_similarity_between_two_numpy_array(data1, data3))
    print("low", get_similarity_between_two_numpy_array(data2, data3))
