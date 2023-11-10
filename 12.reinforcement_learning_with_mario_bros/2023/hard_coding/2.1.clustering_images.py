import os

import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from auto_everything.disk import Disk
disk = Disk()

from auto_everything.cryptography import Password_Generator
password_generator = Password_Generator(
    base_secret_string="yingshaoxo is the strongest person in this world.")


current_folder = disk.get_directory_path(__file__)

source_image_folder = disk.join_paths(current_folder, "raw_images")
disk.create_a_folder(source_image_folder)

target_image_folder = disk.join_paths(current_folder, "seperated_classes")
disk.create_a_folder(target_image_folder)

source_image_path_list = disk.get_files(source_image_folder, type_limiter=[".jpg", ".png"])


model = MobileNetV2(weights='imagenet', include_top=False)


def find_clusters(image_path, save_cluster_path):
    mobilenet_feature_list = []

    for image_path in source_image_path_list:
        img = load_img(image_path, target_size=(224, 224))
        img_data = img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        mobilenet_feature = model.predict(img_data)
        mobilenet_feature_np = np.array(mobilenet_feature)
        mobilenet_feature_list.append(mobilenet_feature_np.flatten())

    mobilenet_feature_list_np = np.array(mobilenet_feature_list)

    kmeans = KMeans(n_clusters=50, random_state=0).fit(mobilenet_feature_list_np)
    y_kmeans = kmeans.labels_
    # dbscan = DBSCAN(eps=0.5, metric='euclidean', min_samples=2).fit(mobilenet_feature_list_np)
    # y_kmeans = dbscan.labels_
    # print(y_kmeans)

    max_labels = max(y_kmeans)
    for i in range(0, len(y_kmeans)):
        if y_kmeans[i] == -1:
            y_kmeans[i] = max_labels+1

    for i, j in zip(source_image_path_list, y_kmeans):
        # print(i, j)
        source_image = i
        target_image = disk.join_paths(save_cluster_path, str(j), disk.get_file_name(i))

        disk.copy_a_file(source_image, target_image)


find_clusters(source_image_folder, target_image_folder)