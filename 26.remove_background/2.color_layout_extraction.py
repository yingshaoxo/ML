# I think I made a mosaic function
from auto_everything.image import Image, get_edge_lines_of_a_image_by_using_yingshaoxo_method
image = Image()

from time import sleep

def real_world_photo_to_animation_graphic_function(a_image, colorful_mode=False, return_a_image_list=False):
    def color_based_shape_layout_extraction(a_image, kernel=5, min_color_distance=50, downscale_ratio=5):
        """
        based on rgb

        yingshaoxo: smooth color check is the key.
        """
        original_image = a_image.copy()
        original_height, original_width = original_image.get_shape()
        a_image = a_image.resize(int(original_height/downscale_ratio), int(original_width/downscale_ratio))

        a_image_backup = a_image.copy()
        a_image = a_image.get_gaussian_blur_image(3, bug_version=True)
        if colorful_mode == True:
            a_image = a_image.get_6_color_simplified_image(free_mode=True)

        height, width = a_image.get_shape()

        step_height = int(height/kernel)
        step_width = int(width/kernel)
        saved_2d_list = [[None]*step_width for _ in range(step_height)]
        layout_image_list = []
        while True:
            done_scan = True
            for row in saved_2d_list:
                for one in row:
                    if one == None:
                        done_scan = False
                        break
                if done_scan == False:
                    break
            if done_scan:
                break

            found_anything_in_this_scan = False
            new_image = a_image.create_an_image(height, width, [0,0,0,0])
            last_rgb = None
            started = False
            temp_rgb_list = []
            last_raw_index = 0
            for height_index in range(step_height):
                found_any_in_this_row = False
                for width_index in range(step_width):
                    start_height = height_index * kernel
                    end_height = start_height + kernel
                    start_width = width_index * kernel
                    end_width = start_width + kernel

                    if saved_2d_list[height_index][width_index] != None:
                        continue

                    temp_sub_image = a_image.raw_data[start_height:end_height]
                    sub_image = []
                    for row in temp_sub_image:
                        sub_image.append(row[start_width: end_width])

                    temp_r = 0
                    temp_g = 0
                    temp_b = 0
                    all_number = 0
                    for row in sub_image:
                        for pixel in row:
                            r,g,b,a = pixel
                            if a == 0:
                                continue
                            temp_r += r
                            temp_g += g
                            temp_b += b
                            all_number += 1
                    if all_number == 0:
                        continue
                    temp_r = temp_r / all_number
                    temp_g = temp_g / all_number
                    temp_b = temp_b / all_number

                    if last_rgb == None:
                        last_rgb = [temp_r, temp_g, temp_b]
                        temp_rgb_list.append(last_rgb)

                    difference = (abs(temp_r - last_rgb[0]) + abs(temp_g - last_rgb[1]) + abs(temp_b - last_rgb[2])) / 3
                    if difference < min_color_distance:
                        # it is same with the last one
                        started = True

                        temp_sub_image_backup = a_image_backup.raw_data[start_height:end_height]
                        sub_image_backup = []
                        for row in temp_sub_image_backup:
                            sub_image_backup.append(row[start_width: end_width])

                        for index, a_row in enumerate(sub_image_backup):
                            new_image.raw_data[start_height+index][start_width: end_width] = a_row[:]
                        saved_2d_list[height_index][width_index] = [start_height, start_width]

                        r_list = [one[0] for one in temp_rgb_list]
                        g_list = [one[1] for one in temp_rgb_list]
                        b_list = [one[2] for one in temp_rgb_list]
                        length = len(r_list)
                        last_rgb = [
                            sum(r_list)/length,
                            sum(g_list)/length,
                            sum(b_list)/length,
                        ]
                        temp_rgb_list.append([temp_r, temp_g, temp_b])

                        found_any_in_this_row = True
                        last_raw_index = height_index
                        found_anything_in_this_scan = True
                    else:
                        continue
                if started == True and found_any_in_this_row == False:
                    break
                if started == True:
                    if (len([one for one in saved_2d_list[height_index] if one != None])/len(saved_2d_list[height_index])) < 0.5:
                        # capture big, let small escape
                        # have bug, it stops when should not 
                        break

            if found_anything_in_this_scan == True:
                # find a way to smooth the edge by using pixel level filting use last_rgb
                new_image.resize(original_height, original_width)
                layout_image_list.append(new_image)
            else:
                break

        return layout_image_list

    def merge_image_layout_list_into_one_image(image_list, pure_color=True):
        base_image = image_list[0]
        height, width = base_image.get_shape()
        base_image = base_image.create_an_image(height, width, [0,0,0,0])
        for a_image in image_list:
            r_all = 0
            g_all = 0
            b_all = 0
            all_numbers = 0
            for row in a_image.raw_data:
                for pixel in row:
                    r,g,b,a = pixel
                    if a != 0:
                        all_numbers += 1
                        r_all += r
                        g_all += g
                        b_all += b
            if all_numbers == 0:
                continue
            r_mean = round(r_all/all_numbers)
            g_mean = round(g_all/all_numbers)
            b_mean = round(b_all/all_numbers)
            for y,row in enumerate(a_image.raw_data):
                for x,pixel in enumerate(row):
                    r,g,b,a = pixel
                    if a != 0:
                        if pure_color == True:
                            base_image.raw_data[y][x] = [r_mean, g_mean, b_mean, 255]
                        else:
                            base_image.raw_data[y][x] = [r,g,b, 255]
        return base_image

    a_image = a_image.copy()
    a_image_list = color_based_shape_layout_extraction(a_image, kernel=5, min_color_distance=50, downscale_ratio=1)
    if return_a_image_list == False:
        a_image = merge_image_layout_list_into_one_image(a_image_list, pure_color=True)
        return a_image
    else:
        return a_image_list

def just_find_one_object_from_center_point(a_image, center_point=None):
    """
    First get the center smallest box average color by doing a resize to 4x4
    Then adjust that box to bigger one by check if the average value changes a lot suddently
    Then do a left to right, up and down scan to find the match rows by using average value from 4x4 box, it is the center object shape
    """
    """
    The basic part of object finding is the same as similar square finding, but have to scan up,down,left,right from center. And in the final view, it scan again for up,down,left,right edge box, use same kernel box average color, but only for filtering out or adding one pixel each time.
    """
    a_image_backup = a_image.copy()
    height, width = a_image.get_shape()
    if center_point == None:
        center_point = [round(height/2), round(width/2)]
    y,x = center_point
    if y < 0 or y >= height or x < 0 or x >= width:
        raise Exception("center_pointer not inside the image")

    hsv_image = a_image.to_hsv()
    #for y in range(height):
    #    for x in range(width):
    #        h,s,v,a = hsv_image[y][x]
    #        s = 255
    #        v = 255
    #        hsv_image[y][x] = [h,s,v,a]
    #hsv_image.to_rgb().save_image_to_file_path("/home/yingshaoxo/Downloads/0.png")
    #exit()

    kernel = round(width/12/2)
    minimum_distance = 40

    # step1, find start box
    history_h = []
    last_usable_average_h = 0
    last_usable_kernel = kernel
    start_height = 0
    end_height = 0
    start_width = 0
    end_width = 0
    while True:
        if len(history_h) >= 2:
            # check if bigger box has an average color that is different than previous one, if so, break the loop
            h1 = history_h[-1]
            h2 = sum(history_h[0:-1])/(len(history_h)-1)
            difference = abs(h1-h2)
            #print("difference:", difference)
            if difference >= minimum_distance:
                last_usable_average_h = h2
                #sub_image = a_image.get_inner_image(start_height, end_height, start_width, end_width)
                #sub_image.save_image_to_file_path("/home/yingshaoxo/Downloads/0.png")
                break
        start_height = y - kernel
        end_height = y + kernel
        start_width = x - kernel
        end_width = x + kernel
        sub_image = a_image.get_inner_image(start_height, end_height, start_width, end_width)
        sub_image.resize(8,8)
        average_h = 0
        counting = 0
        for row in sub_image.raw_data:
            for pixel in row:
                h,s,v,a = pixel
                if a != 0:
                    average_h += h
                    counting += 1
        if counting > 0:
            average_h = average_h/counting
        else:
            # should never happen
            break
        history_h.append(average_h)
        last_usable_kernel = kernel
        kernel = int(kernel * 1.1)
        #print("kernel:", kernel)

    # step2, move box up, down, left, right to match more area. (better find the direction of main h area, only check those direcitons. you do it by check up,down,left,right 1/10 area h value)
    # we only check each box once, so we need to have a cache matrix
    # unsure: it is like a 1x1 to 3x3 to 5x5 process, for each time, we only check around 1 more box.

    # go up,down,left,right to find boundary, to generate a 2d box check table, to filter out some box
    box_length = last_usable_kernel*2
    scale_up_kernel_ratio = 1
    while True:
        new_height = start_height - box_length*scale_up_kernel_ratio
        if new_height < 0:
            break
        scale_up_kernel_ratio += 1
    scale_down_kernel_ratio = 1
    while True:
        new_height = end_height + box_length*scale_down_kernel_ratio
        if new_height > height:
            break
        scale_down_kernel_ratio += 1
    scale_left_kernel_ratio = 1
    while True:
        new_width = start_width - box_length*scale_left_kernel_ratio
        if new_width < 0:
            break
        scale_left_kernel_ratio += 1
    scale_right_kernel_ratio = 1
    while True:
        new_width = end_width + box_length*scale_right_kernel_ratio
        if new_width > width:
            break
        scale_right_kernel_ratio += 1
    all_start_y = start_height - box_length*scale_up_kernel_ratio
    all_end_y = end_height + box_length*scale_down_kernel_ratio
    all_start_x = start_width - box_length*scale_left_kernel_ratio
    all_end_x = end_width + box_length*scale_right_kernel_ratio

    step_height = int((all_end_y - all_start_y) / box_length)
    step_width = int((all_end_x - all_start_x) / box_length)
    checked_box_2d_list = [[None]*step_width for _ in range(step_height)]
    for y in range(step_height):
        for x in range(step_width):
            start_y = y * box_length
            end_y = start_y + box_length
            start_x = x * box_length
            end_x = start_x + box_length

            sub_image = a_image.get_inner_image(start_y, end_y, start_x, end_x)
            sub_image.resize(8,8)
            average_h = 0
            counting = 0
            for row in sub_image.raw_data:
                for pixel in row:
                    h,s,v,a = pixel
                    if a != 0:
                        average_h += h
                        counting += 1
            if counting > 0:
                average_h = average_h/counting
                difference = abs(average_h - last_usable_average_h)
                if difference >= minimum_distance:
                    # not related box
                    checked_box_2d_list[y][x] = None
                else:
                    # related box
                    checked_box_2d_list[y][x] = [start_y, end_y, start_x, end_x]
            else:
                # should filter out since it is all transparent
                checked_box_2d_list[y][x] = None

    new_image = a_image.create_an_image(height, width, [0,0,0,0])
    for row in checked_box_2d_list:
        for one in row:
            if one != None:
                start_y, end_y, start_x, end_x = one
                sub_image = a_image_backup.get_inner_image(start_y, end_y, start_x, end_x)
                new_image.paste_image_on_top_of_this_image(sub_image, start_y, start_x, end_y - start_y, end_x - start_x)
    new_image.save_image_to_file_path("/home/yingshaoxo/Downloads/0.png")


    # step3, in matched box, use smaller 0.5 times box to matched smaller box, filter out other part


def layout_extraction_based_on_color(a_image, kernel=5, minimum_color_distance=5.5):
    # You can use single pixel color, but it may not accurate
    # that's why we use box
    a_image = a_image.copy()
    height, width = a_image.get_shape()

    step_height = int(height/kernel)
    step_width = int(width/kernel)
    layout_list = []
    for y in range(step_height):
        for x in range(step_width):
            start_height = y * kernel
            end_height = start_height + kernel
            start_width = x * kernel
            end_width = start_width + kernel

            sub_image = a_image.get_inner_image(start_height, end_height, start_width, end_width)

            all_r, all_g, all_b = 0,0,0
            counting = 0
            for row in sub_image.raw_data:
                for pixel in row:
                    r,g,b,a = pixel
                    if a != 0:
                        all_r += r
                        all_g += g
                        all_b += b
                        counting += 1
            if counting != 0:
                r = min(max(round(all_r/counting),0),255)
                g = min(max(round(all_g/counting),0),255)
                b = min(max(round(all_b/counting),0),255)
                a = 255
            else:
                #r,g,b,_ = a_image.raw_data[y][x]
                r,g,b,a = [0,0,0,0]

            found_match_layout = False
            for layout_index, one_layout in enumerate(layout_list):
                for box in one_layout:
                    shape, average_rgb = box
                    box_r,box_g,box_b,box_a = average_rgb
                    difference = (abs(r-box_r) + abs(g-box_g) + abs(b-box_b)) / 3
                    if difference <= minimum_color_distance:
                        found_match_layout = True
                        layout_list[layout_index].append(
                            [[start_height,end_height,start_width,end_width], [r,g,b,a]]
                        )
                        break
                if found_match_layout == True:
                    break
            if found_match_layout == False:
                layout_list.append([
                    [[start_height,end_height,start_width,end_width], [r,g,b,a]]
                ])

    new_image = a_image.create_an_image(height, width, [0,0,0,0])
    another_average_color_list = []
    for one_layout in layout_list:
        r_all, g_all, b_all = 0,0,0
        counting = 0
        for box in one_layout:
            shape, average_rgb = box
            start_height,end_height,start_width,end_width = shape

            sub_image = a_image.get_inner_image(start_height, end_height, start_width, end_width)
            all_r, all_g, all_b = 0,0,0
            counting = 0
            for row in sub_image.raw_data:
                for pixel in row:
                    r,g,b,a = pixel
                    if a != 0:
                        all_r += r
                        all_g += g
                        all_b += b
                        counting += 1
        if counting == 0:
            another_average_color_list.append([0,0,0,0])
        else:
            r = min(max(round(all_r/counting),0),255)
            g = min(max(round(all_g/counting),0),255)
            b = min(max(round(all_b/counting),0),255)
            a = 255
            another_average_color_list.append([r,g,b,a])

    for index, one_layout in enumerate(layout_list):
        for box in one_layout:
            shape, average_rgb = box
            start_height,end_height,start_width,end_width = shape
            for y1 in range(start_height, end_height):
                for x1 in range(start_width, end_width):
                    if y1 < 0 or y1 >= height or x1 < 0 or x1 >= width:
                        continue
                    new_image.raw_data[y1][x1] = another_average_color_list[index]

    #print()
    #print(len(layout_list), len(layout_list[-1]))

    return new_image

def layout_extraction_based_on_mobilenet(a_image, kernel=5, minimum_color_distance=5.5, minimum_similarity=0.9):
    # You can use single pixel color, but it may not accurate
    # that's why we use box
    from _5_distinguish_two_sub_image_by_using_deep_learning import get_mobilenet_model, get_image_feature_vector_data, get_similarity_between_two_numpy_array

    a_image = a_image.copy()
    height, width = a_image.get_shape()
    model = get_mobilenet_model("")

    step_height = int(height/kernel)
    step_width = int(width/kernel)
    layout_list = []
    for y in range(step_height):
        for x in range(step_width):
            start_height = y * kernel
            end_height = start_height + kernel
            start_width = x * kernel
            end_width = start_width + kernel

            sub_image = a_image.get_inner_image(start_height, end_height, start_width, end_width)

            all_r, all_g, all_b = 0,0,0
            counting = 0
            for row in sub_image.raw_data:
                for pixel in row:
                    r,g,b,a = pixel
                    if a != 0:
                        all_r += r
                        all_g += g
                        all_b += b
                        counting += 1
            if counting != 0:
                r = min(max(round(all_r/counting),0),255)
                g = min(max(round(all_g/counting),0),255)
                b = min(max(round(all_b/counting),0),255)
                a = 255
            else:
                #r,g,b,_ = a_image.raw_data[y][x]
                r,g,b,a = [0,0,0,0]

            sub_image_hash_list = get_image_feature_vector_data(model, sub_image)

            found_match_layout = False
            for layout_index, one_layout in enumerate(layout_list):
                for box in one_layout:
                    shape, average_rgb, feature_vector = box
                    box_r,box_g,box_b,box_a = average_rgb
                    #difference = (abs(r-box_r) + abs(g-box_g) + abs(b-box_b)) / 3
                    #if difference <= minimum_color_distance:
                    similarity = get_similarity_between_two_numpy_array(sub_image_hash_list, feature_vector)
                    if similarity >= minimum_similarity:
                        found_match_layout = True
                        layout_list[layout_index].append(
                            [[start_height,end_height,start_width,end_width], [r,g,b,a], sub_image_hash_list]
                        )
                        break
                if found_match_layout == True:
                    break
            if found_match_layout == False:
                layout_list.append([
                    [[start_height,end_height,start_width,end_width], [r,g,b,a], sub_image_hash_list]
                ])

    new_image = a_image.create_an_image(height, width, [0,0,0,0])
    another_average_color_list = []
    for one_layout in layout_list:
        r_all, g_all, b_all = 0,0,0
        counting = 0
        for box in one_layout:
            shape, average_rgb,_ = box
            start_height,end_height,start_width,end_width = shape

            sub_image = a_image.get_inner_image(start_height, end_height, start_width, end_width)
            all_r, all_g, all_b = 0,0,0
            counting = 0
            for row in sub_image.raw_data:
                for pixel in row:
                    r,g,b,a = pixel
                    if a != 0:
                        all_r += r
                        all_g += g
                        all_b += b
                        counting += 1
        if counting == 0:
            another_average_color_list.append([0,0,0,0])
        else:
            r = min(max(round(all_r/counting),0),255)
            g = min(max(round(all_g/counting),0),255)
            b = min(max(round(all_b/counting),0),255)
            a = 255
            print(r,g,b)
            another_average_color_list.append([r,g,b,a])

    for index, one_layout in enumerate(layout_list):
        for box in one_layout:
            shape, average_rgb, _ = box
            start_height,end_height,start_width,end_width = shape
            for y1 in range(start_height, end_height):
                for x1 in range(start_width, end_width):
                    if y1 < 0 or y1 >= height or x1 < 0 or x1 >= width:
                        continue
                    new_image.raw_data[y1][x1] = another_average_color_list[index]

    print()
    print(len(layout_list), len(layout_list[-1]))

    return new_image


#a_image_list = color_based_shape_layout_extraction2(a_image, kernel=5, min_color_distance=50, downscale_ratio=1)
#a_image = merge_image_layout_list_into_one_image(a_image_list, pure_color=True)

#source_image_path = "/home/yingshaoxo/Downloads/pillow.bmp"
#source_image_path = "/home/yingshaoxo/Downloads/has_human.bmp"
#source_image_path = "/home/yingshaoxo/Downloads/2286.bmp"
#source_image_path = "/home/yingshaoxo/Downloads/hero3_hqx3.png"
source_image_path = "/home/yingshaoxo/Downloads/a_girl.bmp"
a_image = image.read_image_from_file(source_image_path)

#a_image = a_image.get_gaussian_blur_image(2, bug_version=False)
#a_image = a_image.get_balanced_image()

a_image = layout_extraction_based_on_color(a_image, kernel=5, minimum_color_distance=5.5)
#a_image = layout_extraction_based_on_mobilenet(a_image, kernel=10, minimum_color_distance=10, minimum_similarity=0.9)

#a_image.print()
a_image.save_image_to_file_path("/home/yingshaoxo/Downloads/0.png")

#print("Started!\n\n")
#just_find_one_object_from_center_point(a_image)
