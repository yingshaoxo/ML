from auto_everything.image import Image,rgb_to_hsv
image = Image()

def make_a_line_between_two_points(point_a, point_b):
    y1, x1 = point_a
    y2, x2 = point_b
    new_list = []
    upper_part = y1 - y2
    lower_part = x1 - x2
    if upper_part == 0:
        # horizontal_line
        for x in range(min(x1, x2), max(x1, x2)):
            new_list.append([y1, x])
    elif lower_part == 0:
        # vertical line
        for y in range(min(y1, y2), max(y1, y2)):
            new_list.append([y, x1])
    else:
        slop = upper_part / lower_part
        for x_index in range(min(x1, x2), max(x1, x2)):
            y_index = round(slop*(x_index-x2) + y2)
            new_list.append([y_index, x_index])
    return new_list

def image_erosion(a_image, reverse=False):
    a_image = a_image.copy()

    rows, cols = a_image.get_shape()
    result = [[[0, 0, 0, 0] for _ in range(cols)] for _ in range(rows)]

    the_a = [0, 0, 0, 255]
    the_b = [0, 0, 0, 0]
    if reverse == True:
        the_a,the_b = the_b,the_a

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if all(a_image.raw_data[i+k][j+l] == the_a for k in [-1, 0, 1] for l in [-1, 0, 1]):
                result[i][j] = the_a
            else:
                result[i][j] = the_b

    a_image.raw_data = result
    return a_image

def image_dilate(image):
    rows = len(image)
    cols = len(image[0])
    result = [[[0, 0, 0, 0] for _ in range(cols)] for _ in range(rows)]

    the_a = [0, 0, 0, 255]
    the_b = [0, 0, 0, 0]
    the_a,the_b = the_b,the_a

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if image[i][j] == the_b:
                result[i-1][j-1] = the_b
                result[i-1][j] = the_b
                result[i-1][j+1] = the_b
                result[i][j-1] = the_b
                result[i][j] = the_b
                result[i][j+1] = the_b
                result[i+1][j-1] = the_b
                result[i+1][j] = the_b
                result[i+1][j+1] = the_b

    return result

def get_edge_lines_of_a_image_by_using_yingshaoxo_method(a_image, min_color_distance=15, downscale_ratio=1):
    """
    yingshaoxo: You can use Canny method, but I think it is hard to understand and implement.

    Need a erosion algorithm in here.
    """
    original_image = a_image.copy()
    original_height, original_width = original_image.get_shape()
    a_image = a_image.resize(int(original_height/downscale_ratio), int(original_width/downscale_ratio))

    old_height, old_width = a_image.get_shape()
    new_image = a_image.create_an_image(old_height, old_width, [0,0,0,0])

    height, width = a_image.get_shape()
    point_list = []
    for row_index in range(height):
        previous_pixel = None
        for column_index in range(width):
            pixel = a_image.raw_data[row_index][column_index]
            if previous_pixel != None:
                color_distance_in_horizontal = (abs(previous_pixel[0]-pixel[0]) + abs(previous_pixel[1]-pixel[1]) + abs(previous_pixel[2]-pixel[2]))/3
                if row_index > 0:
                    upper_pixel = a_image.raw_data[row_index-1][column_index]
                    color_distance_in_vertical = (abs(upper_pixel[0]-pixel[0]) + abs(upper_pixel[1]-pixel[1]) + abs(upper_pixel[2]-pixel[2])) / 3
                else:
                    color_distance_in_vertical = 0
                if color_distance_in_horizontal >= min_color_distance or color_distance_in_vertical >= min_color_distance:
                    a_point = [row_index, column_index]
                    point_list.append(a_point)
            previous_pixel = pixel

    for y,x in point_list:
        new_image.raw_data[y][x] = [0,0,0,255]

    new_image.resize(original_height, original_width)
    return new_image

def get_edge_lines_of_a_image_by_using_yingshaoxo_method_use_previous_data(a_image, min_color_distance=15, downscale_ratio=1):
    """
    yingshaoxo: You can use Canny method, but I think it is hard to understand and implement.

    Need a erosion algorithm in here.
    """
    original_image = a_image.copy()
    original_height, original_width = original_image.get_shape()
    a_image = a_image.resize(int(original_height/downscale_ratio), int(original_width/downscale_ratio))

    old_height, old_width = a_image.get_shape()
    new_image = a_image.create_an_image(old_height, old_width, [0,0,0,0])

    previous_kernel = 3
    height, width = a_image.get_shape()
    point_list = []
    for row_index in range(height):
        previous_pixel = None
        for column_index in range(width):
            pixel = a_image.raw_data[row_index][column_index]
            if previous_pixel != None:
                previous_pixel_list = a_image.raw_data[row_index][column_index-previous_kernel:column_index]
                if len(previous_pixel_list) != 0:
                    average_r = sum([one[0] for one in previous_pixel_list])/len(previous_pixel_list)
                    average_g = sum([one[1] for one in previous_pixel_list])/len(previous_pixel_list)
                    average_b = sum([one[2] for one in previous_pixel_list])/len(previous_pixel_list)
                    color_distance_in_horizontal = (abs(average_r-pixel[0]) + abs(average_g-pixel[1]) + abs(average_b-pixel[2]))/3
                else:
                    color_distance_in_horizontal = (abs(previous_pixel[0]-pixel[0]) + abs(previous_pixel[1]-pixel[1]) + abs(previous_pixel[2]-pixel[2]))/3
                if row_index > 0:
                    previous_pixel_list_rows = a_image.raw_data[row_index-previous_kernel:row_index]
                    previous_pixel_list = [one[column_index] for one in previous_pixel_list_rows]
                    if len(previous_pixel_list) > 0:
                        average_r = sum([one[0] for one in previous_pixel_list])/len(previous_pixel_list)
                        average_g = sum([one[1] for one in previous_pixel_list])/len(previous_pixel_list)
                        average_b = sum([one[2] for one in previous_pixel_list])/len(previous_pixel_list)
                        color_distance_in_vertical = (abs(average_r-pixel[0]) + abs(average_g-pixel[1]) + abs(average_b-pixel[2]))/3
                    else:
                        upper_pixel = a_image.raw_data[row_index-1][column_index]
                        color_distance_in_vertical = (abs(upper_pixel[0]-pixel[0]) + abs(upper_pixel[1]-pixel[1]) + abs(upper_pixel[2]-pixel[2])) / 3
                else:
                    color_distance_in_vertical = 0
                if color_distance_in_horizontal >= min_color_distance or color_distance_in_vertical >= min_color_distance:
                    a_point = [row_index, column_index]
                    point_list.append(a_point)
            previous_pixel = pixel

    for y,x in point_list:
        new_image.raw_data[y][x] = [0,0,0,255]

    new_image.resize(original_height, original_width)
    return new_image


def get_edge_lines_of_a_image_by_using_square(a_image, kernel=5, min_color_distance=15, downscale_ratio=1):
    a_image = a_image.copy()
    old_height, old_width = a_image.get_shape()

    a_image = a_image.resize(old_height//downscale_ratio, old_width//downscale_ratio)
    height, width = a_image.get_shape()

    new_image = a_image.create_an_image(height, width, [0,0,0,0])
    #a_image = a_image.get_gaussian_blur_image(2, bug_version=False)
    a_image = a_image.get_balanced_image()

    step_height = int(height/kernel)
    step_width = int(width/kernel)
    average_value_2d_list = [[None] * step_width for one in range(step_height)]
    line_2d_list = [[None] * step_width for one in range(step_height)]
    for y in range(step_height):
        for x in range(step_width):
            start_y = y * kernel
            end_y = start_y + kernel
            start_x = x * kernel
            end_x = start_x + kernel

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
                a = 255
            else:
                #r,g,b,_ = a_image.raw_data[y][x]
                r,g,b,a = [0,0,0,0]

            average_value_2d_list[y][x] = [r,g,b,a]

    for y in range(step_height):
        for x in range(step_width):
            pixel = average_value_2d_list[y][x]

            if y >= 1:
                upper_pixel = average_value_2d_list[y-1][x]
                color_distance_in_vertical = (abs(upper_pixel[0]-pixel[0]) + abs(upper_pixel[1]-pixel[1]) + abs(upper_pixel[2]-pixel[2])) / 3
            else:
                color_distance_in_vertical = 0
            if x >= 1:
                left_pixel = average_value_2d_list[y][x-1]
                color_distance_in_horizontal = (abs(left_pixel[0]-pixel[0]) + abs(left_pixel[1]-pixel[1]) + abs(left_pixel[2]-pixel[2]))/3
            else:
                color_distance_in_horizontal = 0

            if color_distance_in_horizontal >= min_color_distance or color_distance_in_vertical >= min_color_distance:
                start_y = y * kernel
                end_y = start_y + kernel
                start_x = x * kernel
                end_x = start_x + kernel
                line_list = []
                if color_distance_in_horizontal >= min_color_distance:
                    start_point = [start_y, start_x]
                    end_point = [start_y, end_x]
                    line_list.append([start_point, end_point])
                if color_distance_in_vertical >= min_color_distance:
                    start_point = [start_y, start_x]
                    end_point = [end_y, start_x]
                    line_list.append([start_point, end_point])
                line_2d_list[y][x] = line_list

    for y in range(step_height):
        for x in range(step_width):
            line_list = line_2d_list[y][x]
            if line_list != None:
                for line in line_list:
                    start_point, end_point = line
                    points = make_a_line_between_two_points(start_point, end_point)
                    for point in points:
                        y1,x1 = point
                        new_image.raw_data[y1][x1] = [0,0,0,a_image.raw_data[y1][x1][3]]

    new_image.resize(old_height, old_width)
    return new_image

#source_image_path = "/home/yingshaoxo/Downloads/dnf_0.bmp"
source_image_path = "/home/yingshaoxo/Downloads/has_human.bmp"
#source_image_path = "/home/yingshaoxo/Downloads/a_girl.bmp"
a_image = image.read_image_from_file(source_image_path).copy()

#a_image = a_image.get_balanced_image()
a_image = a_image.get_gaussian_blur_image(2, bug_version=False)

#a_image = get_edge_lines_of_a_image_by_using_yingshaoxo_method(a_image, min_color_distance=3, downscale_ratio=1)
#a_image = get_edge_lines_of_a_image_by_using_yingshaoxo_method_use_previous_data(a_image, min_color_distance=3, downscale_ratio=1)
#a_image = image_erosion(a_image, reverse=True)

#a_image = get_edge_lines_of_a_image_by_using_square(a_image, kernel=2, min_color_distance=15, downscale_ratio=1)
a_image = a_image.get_simplified_image_based_on_mean_square_and_edge_line(fill_transparent=False)

a_image.print()
a_image.save_image_to_file_path("/home/yingshaoxo/Downloads/1.png")
