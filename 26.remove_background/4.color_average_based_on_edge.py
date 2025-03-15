from auto_everything.image import Image,single_pixel_hsv_to_rgb, single_pixel_rgb_to_hsv, single_pixel_to_6_main_type_color,get_edge_lines_of_a_image_by_using_yingshaoxo_method
image = Image()

"""
The final goal of this script is to check if two image are the same, for example, in a photo, I think two inner image are the same, they are white, but computer or rgb color doesn't think so. We need to fix this problem.
"""

def get_all_standard_rgb_value():
    "play RGB/hsv color wheel to find the rules"
    pass

def image_erosion(image):
    old_image = image.copy()
    image = old_image.raw_data

    rows = len(image)
    cols = len(image[0])
    result = [[[0,0,0,0] for _ in range(cols)] for _ in range(rows)]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if all(image[i+k][j+l] == [0, 0, 0, 255] for k in [-1, 0, 1] for l in [-1, 0, 1]):
                result[i][j] = [0, 0, 0, 255]

    old_image.raw_data = image
    return old_image

def get_simplified_image_by_using_mean_square_and_edge_line(a_image, downscale_ratio=1, fill_transparent=False, pre_process=False):
    """
    You could do the mean for each pixel by using "scale up until edge line", but that speed is very slow.
    You can also use circle than square, it is more accurate.
    """
    a_image = a_image.copy()
    old_height, old_width = a_image.get_shape()

    a_image = a_image.resize(old_height//downscale_ratio, old_width//downscale_ratio)
    height, width = a_image.get_shape()

    new_image = a_image.create_an_image(height, width, [0,0,0,0])

    if pre_process == True:
        a_image = a_image.get_gaussian_blur_image(2, bug_version=False)
        a_image = a_image.get_balanced_image()

    if pre_process == True:
        edge_image = a_image.to_edge_line(downscale_ratio=2)
    else:
        edge_image = a_image.to_edge_line(downscale_ratio=1)

    for kernel in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 50, 100]:
        step_height = int(height/kernel)
        step_width = int(width/kernel)
        for y in range(step_height):
            for x in range(step_width):
                start_y = y * kernel
                end_y = start_y + kernel
                start_x = x * kernel
                end_x = start_x + kernel

                ok_for_mean = True
                edge_sub_image = edge_image.get_inner_image(start_y, end_y, start_x, end_x)
                for row in edge_sub_image.raw_data:
                    for r,g,b,a in row:
                        if a == 255:
                            ok_for_mean = False
                            break
                    if ok_for_mean == False:
                        break

                if ok_for_mean == True:
                    sub_image = a_image.get_inner_image(start_y, end_y, start_x, end_x)
                    all_r, all_g, all_b, _ = 0,0,0,0
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
                    else:
                        r,g,b,_ = a_image.raw_data[y][x]

                    for y1 in range(start_y, end_y):
                        for x1 in range(start_x, end_x):
                            if y1 < 0 or y1 >= height or x1 < 0 or x1 >= width:
                                continue
                            new_image.raw_data[y1][x1] = [r,g,b,a_image.raw_data[y1][x1][3]]

    if fill_transparent == True:
        kernel = 5
        step_height = int(height/kernel)
        step_width = int(width/kernel)
        for y in range(step_height):
            for x in range(step_width):
                start_y = y * kernel
                end_y = start_y + kernel
                start_x = x * kernel
                end_x = start_x + kernel

                sub_image = new_image.get_inner_image(start_y, end_y, start_x, end_x)
                all_r, all_g, all_b, _ = 0,0,0,0
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
                else:
                    r,g,b,_ = a_image.raw_data[y][x]

                for y1 in range(start_y, end_y):
                    for x1 in range(start_x, end_x):
                        if y1 < 0 or y1 >= height or x1 < 0 or x1 >= width:
                            continue
                        if new_image.raw_data[y1][x1][3] != 0:
                            continue
                        new_image.raw_data[y1][x1] = [r,g,b,255]

    new_image.resize(old_height, old_width)
    return new_image

def just_find_one_object_from_center_point(a_image, center_point=None):
    """
    First get the center smallest box average color by doing a resize to 4x4
    Then adjust that box to bigger one by check if the average value changes a lot suddently
    Then do a left to right, up and down scan to find the match rows by using average value from 4x4 box, it is the center object shape
    """
    a_image = a_image.copy()
    height, width = a_image.get_shape()
    if center_point == None:
        center_point = [round(height/2), round(width/2)]
    y,x = center_point
    if y < 0 or y >= height or x < 0 or x >= width:
        raise Exception("center_pointer not inside the image")

    hsv_image = a_image.to_hsv()

    kernel = round(width/12/2)
    minimum_distance = 10 #12

    # step1, find start box
    history_h = []
    history_rgb = []
    last_usable_average_h = 0
    last_usable_average_rgb = [0,0,0]
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
            length = len(history_rgb) - 1
            r1,g1,b1 = history_rgb[-1]
            r2,g2,b2 = [
                sum([one[0] for one in history_rgb[0:-1]])/length,
                sum([one[1] for one in history_rgb[0:-1]])/length,
                sum([one[2] for one in history_rgb[0:-1]])/length,
            ]
            difference2 = (abs(r1-r2) + abs(g1-g2) + abs(b1-b2))/3
            if difference >= minimum_distance or difference2 >= minimum_distance:
                last_usable_average_h = h2
                last_usable_average_rgb = [r2,g2,b2]

                #sub_image = a_image.get_inner_image(start_height, end_height, start_width, end_width)
                #sub_image.save_image_to_file_path("/home/yingshaoxo/Downloads/1.png")
                #exit()
                break
        start_height = y - kernel
        end_height = y + kernel
        start_width = x - kernel
        end_width = x + kernel
        if start_height < 0 or end_height >= height or start_width < 0 or end_width >= width:
            length = len(history_rgb) - 1
            h2 = sum(history_h[0:-1])/length
            r2,g2,b2 = [
                sum([one[0] for one in history_rgb[0:-1]])/length,
                sum([one[1] for one in history_rgb[0:-1]])/length,
                sum([one[2] for one in history_rgb[0:-1]])/length,
            ]
            last_usable_average_h = h2
            last_usable_average_rgb = [r2,g2,b2]
            break
        sub_image = hsv_image.get_inner_image(start_height, end_height, start_width, end_width)
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

        sub_image2 = a_image.get_inner_image(start_height, end_height, start_width, end_width)
        sub_image2.resize(8,8)
        average_r, average_g, average_b = 0,0,0
        counting = 0
        for row in sub_image2.raw_data:
            for pixel in row:
                r,g,b,a = pixel
                if a != 0:
                    average_r += r
                    average_g += g
                    average_b += b
                    counting += 1
        if counting > 0:
            average_r = average_r/counting
            average_g = average_g/counting
            average_b = average_b/counting
        else:
            # should never happen
            break
        history_rgb.append([average_r, average_g, average_b])

        last_usable_kernel = kernel
        kernel = int(kernel * 1.1)
        #print("kernel:", kernel)

    return [start_height,end_height,start_width,end_width], last_usable_average_rgb

#def get_simplified_image_by_using_increasing_square_kernel(a_image, downscale_ratio=1):
#    a_image = a_image.copy()
#    old_height, old_width = a_image.get_shape()
#
#    a_image = a_image.resize(old_height//downscale_ratio, old_width//downscale_ratio)
#    height, width = a_image.get_shape()
#
#    new_image = a_image.create_an_image(height, width, [0,0,0,0])
#    a_image = a_image.get_gaussian_blur_image(2, bug_version=False)
#    a_image = a_image.get_balanced_image()
#
#    for kernel_index, kernel in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 50, 100]):
#        mininum_color_distance = 19 - kernel_index
#        step_height = int(height/kernel)
#        step_width = int(width/kernel)
#        for y in range(step_height):
#            for x in range(step_width):
#                start_y = y * kernel
#                end_y = start_y + kernel
#                start_x = x * kernel
#                end_x = start_x + kernel
#
#                sub_image = a_image.get_inner_image(start_y, end_y, start_x, end_x)
#                all_r, all_g, all_b, _ = 0,0,0,0
#                counting = 0
#                for row in sub_image.raw_data:
#                    for pixel in row:
#                        r,g,b,a = pixel
#                        if a != 0:
#                            all_r += r
#                            all_g += g
#                            all_b += b
#                            counting += 1
#                if counting != 0:
#                    r = min(max(round(all_r/counting),0),255)
#                    g = min(max(round(all_g/counting),0),255)
#                    b = min(max(round(all_b/counting),0),255)
#                else:
#                    #r,g,b,_ = a_image.raw_data[y][x]
#                    r,g,b = [0,0,0]
#
#                # should compare its arounding 9x9 squares, eat similar box, let similar box use same average color
#
#                for y1 in range(start_y, end_y):
#                    for x1 in range(start_x, end_x):
#                        new_image.raw_data[y1][x1] = [r,g,b,a_image.raw_data[y1][x1][3]]
#
#    new_image.resize(old_height, old_width)
#    return new_image

source_image_path = "/home/yingshaoxo/Downloads/pillow.bmp"
#source_image_path = "/home/yingshaoxo/Downloads/has_human.bmp"
#source_image_path = "/home/yingshaoxo/Downloads/2286.bmp"
#source_image_path = "/home/yingshaoxo/Downloads/dnf_0.bmp"
#source_image_path = "/home/yingshaoxo/Downloads/no_human.bmp"
#source_image_path = "/home/yingshaoxo/Downloads/food.bmp"
#source_image_path = "/home/yingshaoxo/Downloads/hero2_simplified.png"
#source_image_path = "/home/yingshaoxo/Downloads/a_girl.bmp"
a_image = image.read_image_from_file(source_image_path)

#a_image = just_find_one_object_from_center_point(a_image)
#a_image = get_simplified_image_by_using_increasing_square_kernel(a_image)

#a_image = get_simplified_image_by_using_mean_square_and_edge_line(a_image, downscale_ratio=1, fill_transparent=False, pre_process=False)

a_image = a_image.get_6_color_simplified_image(balance=True, free_mode=True, animation_mode=False, accurate_mode=False, kernel=11)

a_image.save_image_to_file_path("/home/yingshaoxo/Downloads/1.png")
