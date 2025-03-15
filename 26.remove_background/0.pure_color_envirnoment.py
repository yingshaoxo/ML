from auto_everything.image import Image
image = Image()

source_image_path = "/home/yingshaoxo/Downloads/no_human.bmp"
no_human = image.read_image_from_file(source_image_path).copy()

source_image_path = "/home/yingshaoxo/Downloads/has_human.bmp"
human_image = image.read_image_from_file(source_image_path).copy()

def remove_background(background_image, has_human_image, kernel=5, compare_number=0.5, complex_mode=True):
    """
    human think a smaller square picture as white one, but computer think it is black one. that's why computer is not accurate.
    """
    height, width = background_image.get_shape()
    kernel = kernel
    all_number = kernel*kernel
    step_height = int(height/kernel)
    step_width = int(width/kernel)
    saved_2d_list = [[False]*step_width for _ in range(step_height)]
    new_image = image.create_an_image(height, width, [0,0,0,0])
    for height_index in range(step_height):
        for width_index in range(step_width):
            start_height = height_index * kernel
            end_height = start_height + kernel
            start_width = width_index * kernel
            end_width = start_width + kernel

            temp_background_sub_image = background_image.raw_data[start_height:end_height]
            background_sub_image = []
            for row in temp_background_sub_image:
                background_sub_image.append(row[start_width: end_width])

            temp_human_sub_image = has_human_image.raw_data[start_height:end_height]
            human_sub_image = []
            for row in temp_human_sub_image:
                human_sub_image.append(row[start_width: end_width])

            background_2 = image.create_an_image(kernel, kernel, [0,0,0,0])
            background_2.raw_data = background_sub_image

            human_2 = image.create_an_image(kernel, kernel, [0,0,0,0])
            human_2.raw_data = human_sub_image

            if background_2.compare(human_2) < compare_number:
                # it is human, not background
                for index, human_row in enumerate(human_sub_image):
                    new_image.raw_data[start_height+index][start_width: end_width] = human_row[:]
                saved_2d_list[height_index][width_index] = True
            else:
                saved_2d_list[height_index][width_index] = False

    if complex_mode == False:
        return new_image

    """
    all_number = (kernel+kernel)**2
    for height_index in range(int(height/kernel)):
        for width_index in range(int(width/kernel)):
            start_height = height_index * kernel
            end_height = start_height + kernel
            start_width = width_index * kernel
            end_width = start_width + kernel

            current_point = saved_2d_list[height_index][width_index]
            if current_point == True:
                continue

            start_height = start_height - kernel
            end_height = end_height + kernel
            start_width = start_width - kernel
            end_width = end_width + kernel

            temp_background_sub_image = background_image.raw_data[start_height:end_height]
            background_sub_image = []
            for row in temp_background_sub_image:
                background_sub_image.append(row[start_width: end_width])

            temp_human_sub_image = has_human_image.raw_data[start_height:end_height]
            human_sub_image = []
            for row in temp_human_sub_image:
                human_sub_image.append(row[start_width: end_width])

            background_2 = image.create_an_image(kernel*2, kernel*2, [0,0,0,0])
            background_2.raw_data = background_sub_image

            human_2 = image.create_an_image(kernel*2, kernel*2, [0,0,0,0])
            human_2.raw_data = human_sub_image

            try:
                similarity = background_2.compare(human_2)
            except Exception as e:
                similarity = 0
            if similarity < 0.7:
                # it is human, not background
                for index, human_row in enumerate(human_sub_image):
                    new_image.raw_data[start_height+index][start_width: end_width] = human_row[:]
    """

    index_2 = 0 #here I can't use 'index' variable because the stupid new python changed index variable at runtime
    for height_index in range(int(height/kernel)):
        for width_index in range(int(width/kernel)):
            start_height = height_index * kernel
            end_height = start_height + kernel
            start_width = width_index * kernel
            end_width = start_width + kernel

            temp_human_sub_image = has_human_image.raw_data[start_height:end_height]
            human_sub_image = []
            for row in temp_human_sub_image:
                human_sub_image.append(row[start_width: end_width])

            current_point = saved_2d_list[height_index][width_index]
            if current_point == True:
                continue

            point_y, point_x = height_index, width_index
            condition_1 = False
            while point_x > 0:
                point_x -= 1
                a_point = saved_2d_list[point_y][point_x]
                if a_point == True:
                    condition_1 = True
                    break
            point_y, point_x = height_index, width_index
            condition_2 = False
            while point_x < step_width-1:
                point_x += 1
                a_point = saved_2d_list[point_y][point_x]
                if a_point == True:
                    condition_2 = True
                    break
            point_y, point_x = height_index, width_index
            condition_3 = False
            while point_y > 0:
                point_y -= 1
                a_point = saved_2d_list[point_y][point_x]
                if a_point == True:
                    condition_3 = True
                    break
            point_y, point_x = height_index, width_index
            condition_4 = False
            while point_y < step_height-1:
                point_y += 1
                a_point = saved_2d_list[point_y][point_x]
                if a_point == True:
                    condition_4 = True
                    break

            ok = False
            if (condition_1 and condition_2 and condition_3 and condition_4):
                ok = True
            #if (condition_1 and condition_2 and condition_3):
            #    ok = True
            #if (condition_2 and condition_3 and condition_4):
            #    ok = True
            #if (condition_3 and condition_4 and condition_1):
            #    ok = True
            #if (condition_4 and condition_1 and condition_2):
            #    ok = True
            #if (condition_3 and condition_4):
            #    ok = True
            if ok == True:
                # it is human, not background
                for index, human_row in enumerate(human_sub_image):
                    new_image.raw_data[start_height+index][start_width: end_width] = human_row[:]

    return new_image

a_image = remove_background(no_human, human_image, kernel=5, compare_number=0.5, complex_mode=True)

a_image.print()
a_image.save_image_to_file_path("/home/yingshaoxo/Downloads/1.png")
