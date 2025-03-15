from auto_everything.image import Image,single_pixel_hsv_to_rgb, single_pixel_rgb_to_hsv, single_pixel_to_6_main_type_color,rgb_to_greyscale,range_map
image = Image()

"""
The final goal of this script is to check if two image are the same, for example, in a photo, I think two inner image are the same, they are white, but computer or rgb color doesn't think so. We need to fix this problem.
"""

def get_all_standard_rgb_value():
    "play RGB/hsv color wheel to find the rules"
    pass

#source_image_path = "/home/yingshaoxo/Downloads/pillow.bmp"
#source_image_path = "/home/yingshaoxo/Downloads/has_human.bmp"
#source_image_path = "/home/yingshaoxo/Downloads/2286.bmp"
#source_image_path = "/home/yingshaoxo/Downloads/dnf_0.bmp"
#source_image_path = "/home/yingshaoxo/Downloads/no_human.bmp"
#source_image_path = "/home/yingshaoxo/Downloads/food.bmp"
#source_image_path = "/home/yingshaoxo/Downloads/hero2_simplified.png"
source_image_path = "/home/yingshaoxo/Downloads/a_girl.bmp"

a_image = image.read_image_from_file(source_image_path)
height, width = a_image.get_shape()

#a_image = a_image.resize(height//2, width//2)
#height, width = a_image.get_shape()

#a_image = a_image.get_balanced_image()
#a_image = a_image.get_gaussian_blur_image(2, bug_version=False)

"""
a_image = a_image.to_hsv()
height, width = a_image.get_shape()
for y in range(height):
    for x in range(width):
        h,s,v,a = a_image.raw_data[y][x]
        a_image.raw_data[y][x] = [h,s,range_map(v, 0,255,55,200),a]
a_image = a_image.to_rgb()
"""

a_image = a_image.get_simplified_image(200, extreme_color_number=127)
#a_image = a_image.get_6_color_simplified_image(balance=True, free_mode=True, animation_mode=True, accurate_mode=False, kernel=11)
#a_image = a_image.to_edge_line(downscale_ratio=1)
#a_image = real_world_photo_to_animation_graphic_function(a_image, colorful_mode=False)

a_image.save_image_to_file_path("/home/yingshaoxo/Downloads/0.png")
a_image.save_image_to_file_path("/home/yingshaoxo/Downloads/0.txt", extreme=False)
