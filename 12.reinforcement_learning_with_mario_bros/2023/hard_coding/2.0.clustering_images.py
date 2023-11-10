from auto_everything.disk import Disk
disk = Disk()

from sewar.full_ref import msssim, uqi
import cv2

current_folder = disk.get_directory_path(__file__)
output_folder = disk.join_paths(current_folder, "seperated_classes")

disk.create_a_folder(output_folder)

#images = disk.get_files(folder=disk.join_paths(current_folder, "raw_seperate_images"))
images = disk.get_files(folder=disk.join_paths(current_folder, "raw_images"))
images_dict = {image_path:cv2.imread(image_path) for image_path in images}

relation_rate = 0.95

classes = {}

index = 0
for image_path_1 in list(images_dict.keys()).copy():
    if image_path_1 not in images_dict.keys():
        continue
    image_1 = images_dict[image_path_1]

    for image_path_2 in list(images_dict.keys()).copy():
        if image_path_2 not in images_dict.keys():
            continue

        if image_path_1 != image_path_2:
            image_2 = images_dict[image_path_2]

            similarity = uqi(image_1, image_2)
            # print(similarity)
            if similarity >= relation_rate:
                # print("found")
                # print(image_path_1)
                # print(image_path_2)
                # exit()
                if index not in classes.keys():
                    classes[index] = []
                
                classes[index].append(image_path_2)
                del images_dict[image_path_2]
    
    if index in classes.keys():
        classes[index].append(image_path_1)
        del images_dict[image_path_1]

        # copy images to related folder 
        for path in classes[index]:
            disk.copy_a_file(path, disk.join_paths(output_folder, str(index), disk.get_file_name(path)) )
        print(f"index {index} done.")

        index += 1
    else:
        # finished
        break

print("Done.")