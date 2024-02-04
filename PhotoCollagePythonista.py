'''
Created on May 24, 2020

@author: Tim Wilson twilsonco @t gmail d.t com
'''
import dialogs
import photos
import console
import appex

import datetime
import time
import random
import math
import os
from PIL import Image, ImageDraw, ImageChops, ImageOps
from operator import itemgetter
from multiprocessing.pool import ThreadPool as Pool

# got idea from https://medium.com/@jtreitz/the-algorithm-for-a-perfectly-balanced-photo-gallery-914c94a5d8af

# start partition problem algorithm from https://stackoverflow.com/a/7942946
# modified to act on list of images rather than the weights themselves
# more info on the partition problem http://www8.cs.umu.se/kurser/TDBAfl/VT06/algorithms/BOOK/BOOK2/NODE45.HTM
"""
Partition a sequence into k sublists with approximately equal sums.

:param seq: The sequence to be partitioned.
:param k: The number of sublists to partition the sequence into.
:param data_list: (optional) A list of data corresponding to the elements in the sequence.

:return: A list of k sublists, each containing elements from the original sequence.

The function partitions the sequence into k sublists such that the sums of the elements in each sublist are approximately equal. If k is less than or equal to 0, an empty list is returned. If k is greater than the length of the sequence, each element of the sequence is returned as a separate sublist.

If a data_list is provided, it should have the same length as the sequence. The elements in the sublists will be selected from the data_list instead of the sequence.

The function uses dynamic programming to find the optimal partitioning. It first calculates a table of partial sums for each prefix of the sequence. Then, it iteratively selects the optimal partitioning by considering all possible splits of the sequence.

Example usage:
    seq = [1, 2, 3, 4, 5]
    k = 3
    result = linear_partition(seq, k)
    # result = [[1, 2], [3, 4], [5]]
"""
def linear_partition(seq, k, data_list = None, do_rotate = False):
    if k <= 0:
        return []
    n = len(seq) - 1
    if k > n:
        return map(lambda x: [x], seq)
    _, solution = linear_partition_table(seq, k)
    k, ans = k-2, []
    if data_list == None or len(data_list) != len(seq):
        while k >= 0:
            row = [[seq[i] for i in range(solution[n-1][k]+1, n+1)]]
            if do_rotate:
                ans += row
            else:
                ans = row + ans
            n, k = solution[n-1][k], k-1
        row = [[seq[i] for i in range(0, n+1)]]
        if do_rotate:
            ans += row
        else:
            ans = row + ans
    else:
        while k >= 0:
            row = [[data_list[i] for i in range(solution[n-1][k]+1, n+1)]]
            if do_rotate:
                ans += row
            else:
                ans = row + ans
            n, k = solution[n-1][k], k-1
        row = [[data_list[i] for i in range(0, n+1)]]
        if do_rotate:
            ans += row
        else:
            ans = row + ans
    return ans

"""
Partition a sequence into k sublists with approximately equal sums.

:param seq: The sequence to be partitioned.
:param k: The number of sublists to partition the sequence into.

:return: A tuple containing two lists: the table and the solution.
The table is a 2D list representing the dynamic programming table used in the algorithm.
The solution is a 2D list representing the optimal partitioning of the sequence.

The function partitions the sequence into k sublists such that the sums of the elements in each sublist are approximately equal. 
The algorithm uses dynamic programming to find the optimal partitioning. It first calculates a table of partial sums for each prefix of the sequence. 
Then, it iteratively selects the optimal partitioning by considering all possible splits of the sequence.

The table is a 2D list with dimensions (n, k), where n is the length of the sequence and k is the number of sublists.
Each element in the table represents the sum of the elements in the corresponding sublist.
The solution is a 2D list with dimensions (n-1, k-1), where each element represents the index of the split point for the corresponding sublist.

Example usage:
    seq = [1, 2, 3, 4, 5]
    k = 3
    table, solution = linear_partition_table(seq, k)
"""
def linear_partition_table(seq, k):
    n = len(seq)
    table = [[0] * k for x in range(n)]
    solution = [[0] * (k-1) for x in range(n-1)]
    for i in range(n):
        table[i][0] = seq[i] + (table[i-1][0] if i else 0)
    for j in range(k):
        table[0][j] = seq[0]
    for i in range(1, n):
        for j in range(1, k):
            table[i][j], solution[i-1][j-1] = min(
                ((max(table[x][j-1], table[i][0]-table[x][0]), x) for x in range(i)),
                key=itemgetter(0))
    return (table, solution)
# end partition problem algorithm

"""
Create batches of indices from a given range.

:param n: The total number of indices.
:param min_batch_size: The minimum size of each batch.
:param max_batch_size: The maximum size of each batch.
:param mid_size_min_factor: (optional) The minimum factor for determining the size of mid-sized batches. Default is 0.7.

:return: A list of batches, where each batch is a list of indices.

The function creates batches of indices from a given range, such that the total number of indices is divided into multiple batches. The size of each batch is randomly determined within the specified minimum and maximum batch sizes. If the remaining number of indices is less than the mid-sized batch size, a single index is assigned to each batch. Otherwise, a random batch size is chosen within the specified range.

The batches are returned in random order. Additionally, the elements of each batch are replaced with consecutive indices starting from 0. For example, the first batch will contain indices [0, 1, 2], the second batch will contain indices [3, 4, 5], and so on.

Example usage:
    n = 10
    min_batch_size = 2
    max_batch_size = 4
    mid_size_min_factor = 0.7
    batches = create_batches(n, min_batch_size, max_batch_size, mid_size_min_factor)
"""
def create_batches(n, min_batch_size, max_batch_size, mid_size_min_factor = 0.7):
    batches = []
    num_remaining = n
    i = 0
    seq = range(n)
    mid_size = int(mid_size_min_factor * min_batch_size + (1 - mid_size_min_factor) * max_batch_size)
    while num_remaining > 0:
        if num_remaining < mid_size:
            batches.append([seq[i]])
            i += 1
            num_remaining -= 1
        else:
            tmp_max_batch_size = min(max_batch_size, num_remaining)
            batch_size = random.randint(min_batch_size, tmp_max_batch_size)
            batches.append(seq[i:i+batch_size])
            i += batch_size
            num_remaining -= batch_size
    # return batches in random order
    random.shuffle(batches)
    # now replace elements of batches so that first batch is [0,1,2], second is [3,4,5], etc.
    i = 0
    out_batches = []
    for batch in batches:
        out_batches.append([i + j for j in range(len(batch))])
        i += len(batch)
    return out_batches

"""
Add rounded corners to an image.

:param im: The image to add rounded corners to.
:param rad: The radius of the rounded corners.

:return: The image with rounded corners.

The function creates a circle image with the specified radius and fills it with white color. It then creates an alpha channel image with the same size as the input image and pastes the circle image onto the alpha channel at the four corners. Finally, it applies the alpha channel to the input image using the putalpha() method.

Adapted from https://stackoverflow.com/a/11291419/2620767

Example usage:
    im = Image.open('image.jpg')
    rad = 20
    result = add_corners(im, rad)
"""
def add_corners(im, corner_perc, supersample=3):
    # Create a larger image for supersampling
    rad = int(min(im.size) * 0.5 * corner_perc / 100.0)
    large_rad = rad * supersample
    large_size = (im.size[0] * supersample, im.size[1] * supersample)

    # Create the circle mask on the larger image
    circle = Image.new('L', (large_rad * 2, large_rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, large_rad * 2, large_rad * 2), fill=255)

    # Create the alpha mask on the larger image
    alpha = Image.new('L', large_size, "white")
    w, h = large_size
    alpha.paste(circle.crop((0, 0, large_rad, large_rad)), (0, 0))
    alpha.paste(circle.crop((0, large_rad, large_rad, large_rad * 2)), (0, h - large_rad))
    alpha.paste(circle.crop((large_rad, 0, large_rad * 2, large_rad)), (w - large_rad, 0))
    alpha.paste(circle.crop((large_rad, large_rad, large_rad * 2, large_rad * 2)), (w - large_rad, h - large_rad))

    # Resize the image and the alpha mask
    im = im.resize(large_size, Image.LANCZOS)
    # alpha = alpha.resize(large_size, Image.LANCZOS)

    if im.mode == 'RGBA':
        # If the image has an alpha layer, combine it with the new alpha layer
        alpha = ImageChops.multiply(alpha, im.split()[3])

    # Apply the alpha mask to the image
    im.putalpha(alpha)

    # Scale down the image
    im = im.resize((im.size[0] // supersample, im.size[1] // supersample), Image.LANCZOS)

    return im

"""
Rotate an image by a random angle within a specified range.

:param img: The image to be rotated.
:param max_deg: The maximum angle in degrees for the rotation.
:param supersample: (optional) The factor by which to supersample the image before rotation. Default is 4.

:return: The rotated image.

The function takes an image and rotates it by a random angle within the specified range. The rotation is performed by first pasting the image onto a larger image to prevent cutting off corners during rotation. The size of the larger image is calculated based on the width, height, and amount of rotation. The image is then rotated by the random angle using the 'rotate' method. Finally, the rotated image is scaled down by the supersample factor.

Example usage:
    img = Image.open('image.jpg')
    max_deg = 45
    supersample = 4
    rotated_img = rotate_image(img, max_deg, supersample)
"""
def rotate_image(img, max_deg, supersample=4):
    # Create a larger image for supersampling
    large_size = (img.size[0] * supersample, img.size[1] * supersample)
    img = img.resize(large_size, Image.LANCZOS)
    # Rotate the image
    rot_angle = random.uniform(-max_deg, max_deg)
    img = img.rotate(rot_angle, expand=True)
    # Scale down the image
    img = img.resize((img.width // supersample, img.height // supersample), Image.LANCZOS)
    return img

def resize_img_to_max_size(img, max_size, no_antialias=False):
    if max_size > 50 and img.width > max_size:
        if no_antialias:
            return img.resize((max_size, int(img.height / img.width * max_size)))
        else:
            return img.resize((max_size, int(img.height / img.width * max_size)), Image.LANCZOS)
    elif img.height > max_size:
        if no_antialias:
            return img.resize((int(img.width / img.height * max_size), max_size))
        else:
            return img.resize((int(img.width / img.height * max_size), max_size), Image.LANCZOS)
    else:
        return img

def clamp(v,l,h):
    return l if v < l else h if v > h else v

# takes list of PIL image objects and returns the collage as a PIL image object
# collage_type can be one of "nested", "row", or "column"
def makeCollage(img_list,
                collage_type = "nested", 
                recursion_depth = 0, 
                max_recursion_depth = 2,
                min_batch_size_in = 3,
                max_batch_size_in = 7,
                spacing = 0, 
                no_antialias = False, 
                background=(0,0,0), 
                target_aspect_ratio = 1.0, 
                max_collage_size = 0, 
                round_image_corners_perc = 0.0,
                round_collage_corners_perc = 0.0,
                rotate_images_max_deg = 0,
                show_recursion_depth = False):
    # check img_ordering and collage_type args
    if collage_type not in ["nested", "rows", "columns"]:
        raise ValueError("collage_type must be one of 'nested', 'rows', or 'columns'")
    
    if collage_type == "nested":
        max_recursion_depth = max(max_recursion_depth, 1)
        
    pool = Pool(1) if appex.is_running_extension() else Pool()

    # perform processing of images for top-level call only
    if recursion_depth == 0:
        # round corners of images as percentage of shortest edge if requested
        if round_image_corners_perc > 0.0:
            def add_corner_to_img(img):
                return add_corners(img, round_image_corners_perc)
            img_list = pool.map(add_corner_to_img, img_list)
        # resize all images so that the longest edge is less than or equal to max_collage_size (if max_collage_size is greater than 0)
        if max_collage_size > 0:
            def resize_to_max(img):
                return resize_img_to_max_size(img, max_collage_size, no_antialias)
            img_list = pool.map(resize_to_max, img_list)
        # rotate images randomly within ±rotate_images_max_deg if requested
        if rotate_images_max_deg > 0:
            def rotate_img(img):
                return rotate_image(img, rotate_images_max_deg)
            img_list = pool.map(rotate_img, img_list)
    
    # for column-major collage, set the do_rotate flag to True. For nested collage, set to false for top-level call and randint(0,1) for recursive calls
    if collage_type == "columns":
        do_rotate = True
    elif collage_type == "nested":
        do_rotate = False if recursion_depth == 0 else random.randint(0,1)
    else:
        do_rotate = False
    
    if do_rotate:
        target_aspect_ratio = 1.0 / target_aspect_ratio
    
    # update max batch size based on max_recursion_depth (thus allowing for more levels of nesting)
    max_batch_size = int(round(clamp(max_batch_size_in * max_recursion_depth, max_batch_size_in, max(min_batch_size_in + 2, len(img_list) / 3))))
    
    # if collage_type is "nested", then we need to recursively call makeCollage on each batch of images.
    # We can only recurse of the following are true:
    # 1. recursion_depth is less than max_recursion_depth
    # 2. the max number of batches is greater than 3
    # 3. the aspect ratio of all images if placed horizontally is less than or equal to target_aspect_ratio
    # 4. the aspect ratio of all images if placed vertically is greater than or equal to target_aspect_ratio
    if collage_type == "nested":
        max_width = max([img.width for img in img_list])
        max_height = max([img.height for img in img_list])
        sum_width = sum([img.width for img in img_list])
        sum_height = sum([img.height for img in img_list])
        ar_if_vertical = max_width / sum_height
        ar_if_horizontal = sum_width / max_height
        max_num_batches = len(img_list) / min_batch_size_in
        do_recurse = True
        if target_aspect_ratio >= ar_if_horizontal:
            do_recurse = False
        if target_aspect_ratio <= ar_if_vertical:
            do_recurse = False
        if recursion_depth >= max_recursion_depth:
            do_recurse = False
        if max_num_batches <= 3:
            do_recurse = False
        if do_recurse:
            # create batches of images
            batches = create_batches(len(img_list), min_batch_size_in, max_batch_size)
            # create collage from each batch, calling makeCollage recursively for batches of len > 1
            tmp_img_list = []
            for batch in batches:
                if len(batch) == 1:
                    img = img_list[batch[0]]
                    tmp_img_list.append(img)
                else:
                    downscale_factor = 0.75
                    # for batches, save memory by reducing image size down by downscale_factor
                    batch_img_list = [img_list[img_num] for img_num in batch]
                    def downscale_img(img):
                        if no_antialias:
                            return img.resize((int(img.width * downscale_factor), int(img.height * downscale_factor)))
                        else:
                            return img.resize((int(img.width * downscale_factor), int(img.height * downscale_factor)), Image.LANCZOS)
                    batch_img_list = pool.map(downscale_img, batch_img_list)
                    # aspect ratio of batch is a random in [0.5, 2]
                    batch_target_aspect_ratio = random.uniform(0.5, 2.0)
                    batch_max_collage_size = int(max_collage_size * downscale_factor)
                    img = makeCollage(batch_img_list, 
                                    collage_type=collage_type,
                                    recursion_depth=recursion_depth + 1, 
                                    max_recursion_depth=max_recursion_depth, 
                                    min_batch_size_in=min_batch_size_in, 
                                    max_batch_size_in=max_batch_size_in, 
                                    spacing=int(spacing / downscale_factor),
                                    no_antialias=no_antialias, 
                                    background=background, 
                                    target_aspect_ratio=batch_target_aspect_ratio, 
                                    max_collage_size=batch_max_collage_size,
                                    show_recursion_depth=show_recursion_depth)
                    tmp_img_list.append(img)
            img_list = tmp_img_list
            
    # show colored border around images if requested, to indicate recursion depth
    if show_recursion_depth and collage_type == "nested":
        border_colors = ["cyan", "orange", "red", "purple", "yellow", "blue", "green", "pink", "brown", "gray", "black", "white"]
        border_width_factor = 0.02 * (recursion_depth + 1)
        border_color = border_colors[recursion_depth % len(border_colors)]
        def add_border(img):
            return ImageOps.expand(img, border=int((img.width + img.height) / 2 * border_width_factor), fill=border_color)
        img_list = pool.map(add_border, img_list)
        
    # rotate images if needed
    if do_rotate:
        def rotate_img(img):
            return img.transpose(Image.ROTATE_90)
        img_list = pool.map(rotate_img, img_list)
    
    # generate the input for the partition problem algorithm
    # need list of aspect ratios and number of rows (partitions)
    num_images = len(img_list)
    total_width = sum([img.width for img in img_list])
    total_height = sum([img.height for img in img_list])
    avg_width = total_width / num_images
    avg_height = total_height / num_images
    target_width = (avg_height + avg_width) / 2 * math.sqrt(num_images * target_aspect_ratio)
    
    num_rows = clamp(int(round(total_width / target_width)), 1, num_images)
    num_cols = int(math.ceil(num_images / num_rows))
    
    # resize images based on number of rows. First get common width or height
    max_width = max([img.width for img in img_list])
    max_height = max([img.height for img in img_list])
    average_aspect_ratio = sum([img.width / img.height for img in img_list]) / len(img_list)
    if num_rows == num_images:
        # resize to common width
        common_width = int(round(max(10, min(max_width, max_collage_size/num_images * average_aspect_ratio))))
    elif num_rows == 1:
        # resize to common height
        common_height = int(round(max(10, min(max_height, max_collage_size/num_images / average_aspect_ratio))))
    else:
        # resize to common height
        common_height = int(round(max(10, min(max_height, max_collage_size/num_rows, max_collage_size / num_cols / average_aspect_ratio))))
    
    # do actual resizing
    if num_rows < num_images:
        # resize to common_height
        def resize_to_common_height(img):
            if no_antialias:
                return img.resize((int(img.width * common_height / img.height), common_height))
            else:
                return img.resize((int(img.width * common_height / img.height), common_height), Image.LANCZOS)
        img_list = pool.map(resize_to_common_height, img_list)
    else:
        # resize to common_width
        def resize_to_common_width(img):
            if no_antialias:
                return img.resize((common_width, int(img.height * common_width / img.width)))
            else:
                return img.resize((common_width, int(img.height * common_width / img.width)), Image.LANCZOS)
        img_list = pool.map(resize_to_common_width, img_list)
    
    if num_rows == 1:
        img_rows = [img_list]
    elif num_rows == num_images:
        img_rows = [[img] for img in img_list]
    else:
        aspect_ratios = [int(img.width / img.height * 100) for img in img_list]
    
        # get nested list of images (each sublist is a row in the collage)
        img_rows = linear_partition(aspect_ratios, num_rows, img_list, do_rotate)
    
    # prepare output image
    if background == (0,0,0):
        background += tuple([0])
    else:
        background += tuple([255])
        
    # first combine into rows, whose images already share the same heights. Each row will be a PIL image object.
    # then resize rows to have the same width, and combine into a single PIL image object
    row_imgs = []
    for row in img_rows:
        row_img = Image.new("RGBA", (sum([img.width + spacing for img in row]) - spacing, row[0].height), background)
        x_pos = 0
        for img in row:
            row_img.paste(img, (x_pos,0))
            x_pos += img.width + spacing
        row_imgs.append(row_img)
        
    # resize rows to have the same width, that of the minimum row width, while keeping the same aspect ratio
    min_row_width = min([img.width for img in row_imgs])
    def resize_row_to_min_width(img):
        if no_antialias:
            return img.resize((min_row_width, int(img.height * min_row_width / img.width)))
        else:
            return img.resize((min_row_width, int(img.height * min_row_width / img.width)), Image.LANCZOS)
    row_imgs = pool.map(resize_row_to_min_width, row_imgs)
    
    # pupulate new image
    row_widths = [img.width for img in row_imgs]
    row_heights = [img.height for img in row_imgs]
    w, h = (row_widths[0], sum(row_heights) + spacing * (num_rows - 1))
    
    # combine rows into output image
    out_img = Image.new("RGBA", (w,h), background)
    y_pos = 0
    for img in row_imgs:
        out_img.paste(img, (0,y_pos))
        y_pos += img.height + spacing
    
    # unrotate images if needed
    if do_rotate:
        out_img = out_img.transpose(Image.ROTATE_270)
        
    # round corners of collage as percentage of shortest edge if requested, but only for top-level call
    if recursion_depth == 0 and round_collage_corners_perc > 0.0:
        out_img = add_corners(out_img, round_collage_corners_perc)
    
    return out_img
    
def get_user_options(args = None):
    import copy
    
    default_args = [
        {
            'name': "Continue with current settings",
            'icon': "✔︎",
            'value': ""
        },{
            'name': "Collage type",
            'description': "Simple or nested collage?",
            'icon': "⊞",
            'value': "Nested",
            'options': {'Nested': "⧈", 'Rows': "⏛", 'Columns': "⎅"}
        },{
            'name': "Image order",
            'description': "Select image ordering",
            'icon': "⇅",
            'value': "Input order",
            'options': {'Input order': "➲", 'Random': "⤨", 'Oldest first': "↧", 'Newest first': "↥"}
        },{
            'name': "Collage aspect ratio",
            'description': "The target aspect ratio of the collage",
            'icon': "⧉",
            'value': 1.0,
            'min': 0.001,
            'max': 1000.0,
            'type': float
        },{
            'name': "Image spacing",
            'description': "The spacing between images in pixels",
            'icon': "╬",
            'value': 0,
            'min': 0,
            'type': int,
            'unit': " px"
        },{
            'name': "Round image corners",
            'description': "Round corners of each image by this percent of the shortest edge",
            'icon': "⎄",
            'value': 0,
            'min': 0,
            'max': 100,
            'type': int,
            'unit': "%"
        },{
            'name': "Round collage corners",
            'description': "Round corners of the collage by this percent of the shortest edge",
            'icon': "⎄",
            'value': 0,
            'min': 0,
            'max': 100,
            'type': int,
            'unit': "%"
        },{
            'name': "Rotate images",
            'description': "Rotate images randomly within this value",
            'icon': "⟲",
            'value': 0,
            'min': 0,
            'max': 90,
            'type': int,
            'unit': " deg"
        },{
            'name': "Image count",
            'description': "The maximum number of images in the collage, useful for when an entire album is selected",
            'icon': "#",
            'value': 100,
            'min': 3,
            'type': int
        },{
            'name': "Collage max width/height",
            'description': "The maximum longest edge (width or height) of the collage in pixels",
            'icon': "⁜",
            'value': 5000,
            'min': 100,
            'type': int,
            'unit': " px"
        },{
            'name': "Initial image longest edge",
            'description': "Set to 0 to disable. Resize images on import so that the longest edge (width or height) is no bigger than the specified value. Used to downscale images before processing to speed up the collage generation and lower memory usage",
            'icon': "⤢",
            'value': 1000,
            'min': 0,
            'type': int
        },{
            'name': "No intermediate antialiasing",
            'description': "Should intermediate antialiasing be disabled to speed up collage generation?",
            'icon': "▦",
            'value': True,
            'type': lambda x: bool(int(x))
        },{
            'name': "Max nesting depth",
            'description': "The maximum number of levels of nesting in the collage",
            'icon': "⧈",
            'value': 1,
            'min': 1,
            'type': int
        },{
            'name': "Show nesting level",
            'description': "Should nesting level of each image in the collage be indicate using colored borders?",
            'icon': "▣",
            'value': False,
            'type': lambda x: bool(int(x))
        },{
            'name': "Preprocess images (to fix crashes)",
            'icon': "⚙︎",
            'value': ""
        },{
            'name': "Continue with current settings",
            'icon': "✔︎",
            'value': ""
        }
    ]
    
    if args is not None:
        tmp_args = copy.deepcopy(default_args)
        for i,(arg,val) in enumerate(args.items()):
            for opt in tmp_args:
                if opt['name'] == arg:
                    tmp_args[i]['value'] = val
                    break
        args = tmp_args
    else:
        args = copy.deepcopy(default_args)
    
    # Loop to keep allowing user to change options
    while True:
        option_menu = []
        for option in args:
            option_str = f"{option['icon']} {option['name']}"
            if len(str(option['value'])) > 0:
                option_str += f": {option['value']}"
            if 'unit' in option:
                option_str += f"{option['unit']}"
            option_menu.append(option_str)
        # present menu to user
        chosen_option = dialogs.list_dialog(title='Photo collage configuration', items=option_menu)
        if chosen_option is None:
            return None
        # get index of chosen option
        chosen_option_index = option_menu.index(chosen_option)
        if "Preprocess" in chosen_option:
            args[chosen_option_index]['value'] = True
            break
        if "Continue" in chosen_option:
            break
        # prepare followup dialog to change chosen option
        # get chosen option dict
        opt = args[chosen_option_index]
        if 'options' in opt:
            # show list of options to choose from
            option_list = [f"{icon} {name}" for name,icon in opt['options'].items()] + ["↺ Go back"]
            chosen_value = dialogs.list_dialog(title=f"{opt['description']}", items=option_list)
            if chosen_value is None:
                return None
            if "Go back" not in chosen_value:
                args[chosen_option_index]['value'] = ' '.join(chosen_value.split()[1:])
        else:
            input_valid = False
            while not input_valid:
                desc = opt['description']
                msg = "Enter new value"
                if 'min' in opt:
                    msg += f" (min: {opt['min']}"
                    if 'max' in opt:
                        msg += f", max: {opt['max']}"
                    msg += f", current: {opt['value']}"
                    if 'unit' in opt:
                        msg += f"{opt['unit']}"
                    msg += ")"
                val = opt['value']
                if type(val) == bool:
                    new_value = bool(dialogs.alert(desc, "", "Yes", "No")-1)
                    if new_value == 2:
                        new_value = None
                    else:
                        new_value = (new_value == 0)
                else:
                    try:
                        new_value = dialogs.input_alert(desc, msg, str(val), "OK")
                    except KeyboardInterrupt:
                        new_value = None
                if new_value is None:
                    return None
                if 'type' in opt and type(new_value) != bool:
                    # try to convert to type
                    try:
                        new_value = opt['type'](new_value.strip())
                    except ValueError as e:
                        dialogs.alert(title="Error", message=f"Invalid value: {e}. Please try again", button1="OK", hide_cancel_button=True)
                        continue
                # clamp to min/max if needed
                if 'min' in opt:
                    new_value = max(new_value, opt['min'])
                if 'max' in opt:
                    new_value = min(new_value, opt['max'])
                args[chosen_option_index]['value'] = new_value
                input_valid = True
    # return simplified arg dict
    out_args = {opt['name']: opt['value'] for opt in args}
    return out_args

def get_photo_selection():
    def get_album(f=photos.get_albums, t="album"):
        album_dict = {a.title: a for a in f()}
        chosen_album = dialogs.list_dialog(title=f"Select {t}", items=list(album_dict.keys()))
        if chosen_album is None:
            return None
        return album_dict[chosen_album]
    	
    def get_smart_album():
        return get_album(f=photos.get_smart_albums, t="smart album")
    	
    def get_moment():
        return get_album(f=photos.get_moments, t="moment")
    
    img_sources = {
    	"Photo library": photos.get_assets,
    	"Photo album": get_album,
    	"Smart photo album": get_smart_album,
    	"A moment": get_moment 
    }
    
    # ask where to get images from
    num_assets = 0
    while num_assets < 3:
        img_source = dialogs.list_dialog(title="Select image source", items=list(img_sources.keys()))
        if img_source is None:
            return None	
        image_assets = img_sources[img_source]()
        if image_assets is None:
            return None
        if type(image_assets) == photos.AssetCollection:
            image_assets = image_assets.assets
        num_assets = len(image_assets)
        if num_assets < 3:
            try:
                dialogs.alert(title="Error: too few images", message="Select an image source with at least 3 images", button1="OK", hide_cancel_button=True)
            except KeyboardInterrupt:
                return None
    use_all_assets = num_assets <= 300
    if use_all_assets:
        try:
            use_all_assets = dialogs.alert(title="Use all images?", message=f"Found {num_assets} images in {img_source}. Use all images or select photos to use?", button1="Yes, use all images", button2="No, make selection", hide_cancel_button=True)
            if use_all_assets == 3:
                return None
        except KeyboardInterrupt:
            return None
        use_all_assets = (use_all_assets == 1)
    if not use_all_assets:
        image_assets = photos.pick_asset(title='Select images for collage', assets=image_assets, multi=True)
        if image_assets is None:
            return None
    
    # only keep photos
    image_assets = [img for img in image_assets if img.media_type == 'image']
    
    if len(image_assets) < 3:
        user_exit = dialogs.alert(title="Photo Collage", message="Select 3 or more images.", button1='OK', button3="Quit", hide_cancel_button=True)
        if user_exit == 3:
            return None
        return get_photo_selection()
    return image_assets
        

def preprocess_images(image_assets):
    is_appex = appex.is_running_extension()
    # get current date in yyyy-MM-dd HH:mm:ss format
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_album_name = f"Preprocessed {len(image_assets)} images - {date_str}"
    prompt = f"Please enter the maximum longest edge length for preprocessing photos, in pixels. The smaller the value, the more photos you'll be able to use for a collage. Processed photos will be saved to a new photo album named {new_album_name}."
    while True:
        try:
            img_size = dialogs.input_alert("Preprocess images", prompt, "1500", "OK")
        except KeyboardInterrupt:
            return
        try:
            img_size = int(img_size)
        except ValueError:
            dialogs.alert("Error", "Invalid value. Please try again", "OK", hide_cancel_button=True)
            continue
        break
    # create new album
    new_album = photos.create_album(new_album_name)
    # preprocess images
    def preprocess_image(f):
        img = f.get_image()
        # Need to explicitly tell PIL to rotate image if EXIF orientation data is present
        exif = img.getexif()
        # Remove all exif tags
        for k in exif.keys():
            if k != 0x0112:
                exif[k] = None
                del exif[k]
        # Put the new exif object in the original image
        new_exif = exif.tobytes()
        try:
            img.info["exif"] = new_exif
            # Rotate the image
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        img = resize_img_to_max_size(img, img_size)
        fbase = f"{random.randint(0,99999):05d}"
        try:
            fname = f"{fbase}.jpg"
            img.convert('RGB')
            img.save(fname, format='JPEG', quality=80)
            asset = photos.create_image_asset(fname)
            new_album.add_assets([asset])
            os.remove(fname)
        except Exception:
            try:
                os.remove(f"{fbase}.jpg")
            except Exception:
                pass
            fname = f"{fbase}.png"
            img.save(fname, format='PNG')
            asset = photos.create_image_asset(fname)
            new_album.add_assets([asset])
            os.remove(fname)
    pool = Pool(1) if is_appex else Pool()
    print(f"Preprocessing {len(image_assets)} images...")
    pool.map(preprocess_image, image_assets)
    print(f"Finished preprocessing {len(image_assets)} images")
    try:
        dialogs.alert("Finished preprocessing images", f"Preprocessed {len(image_assets)} images. Processed photos saved to new album named {new_album_name}. Re-run this program using those images.", "OK", hide_cancel_button=True)
    except KeyboardInterrupt:
        pass

# modified (significantly) from https://github.com/delimitry/collage_maker
# this main function is for the CLI implementation
def main():
    console.clear()
    print_output = True
    user_exit = 0
    args = None
    PIL_images = [] #appex.get_images()
    is_appex = appex.is_running_extension()
    if is_appex:
        print_output = False
    if len(PIL_images) == 0:
        PIL_images = []
    else:
        if len(PIL_images) < 3:
            dialogs.alert("Must provide 3 or more images", button1="OK", hide_cancel_button=True)
            return
    
    while user_exit <= 1:
        # get images
        if len(PIL_images) == 0:
            image_assets = get_photo_selection()
            if image_assets is None:
                return
        else:
            image_assets = []
        
        while user_exit <= 2 and (len(image_assets) > 2 or len(PIL_images) > 2):
            # get user options
            args = get_user_options(args)
            if args is None:
                return
            
            # preprocess images if requested
            if args['Preprocess images (to fix crashes)'] == True:
                preprocess_images(image_assets)
                return
            
            start_time = time.time()
            
            # apply image ordering
            if args['Image order'] == "Random":
                random.shuffle(image_assets)
            elif args['Image order'] == "Oldest first":
                image_assets.sort(key=lambda x: x.creation_date.timestamp())
            elif args['Image order'] == "Newest first":
                image_assets.sort(key=lambda x: x.creation_date.timestamp(), reverse=True)
            
            if print_output:
                print(f"Found {len(image_assets)} images")
            
            # apply image count limit
            if args['Image count'] < len(image_assets):
                image_assets = image_assets[:args['Image count']]
            
            if print_output:
                print(f"Using {len(image_assets)} images")
            
            # get PIL image objects for all the photos
            if user_exit <= 1 or len(PIL_images) != len(image_assets):
                if print_output:
                    print('Loading photos...')
                def load_PIL_image(f):
                    img = f.get_image()
                    # Need to explicitly tell PIL to rotate image if EXIF orientation data is present
                    exif = img.getexif()
                    # Remove all exif tags
                    for k in exif.keys():
                        if k != 0x0112:
                            exif[k] = None
                            del exif[k]
                    # Put the new exif object in the original image
                    new_exif = exif.tobytes()
                    try:
                        img.info["exif"] = new_exif
                        # Rotate the image
                        img = ImageOps.exif_transpose(img)
                    except Exception:
                        pass
                    return resize_img_to_max_size(img, args['Initial image longest edge'], args['No intermediate antialiasing'])
                pool = Pool(1) if is_appex else Pool()
                PIL_images = pool.map(load_PIL_image, image_assets)
                    
                image_list = [(i,image_assets[i],PIL_images[i]) for i in range(len(PIL_images))]
            
            # apply image ordering (again, in case it's a rerun of the collage)
            if args['Image order'] == "Random":
                random.shuffle(image_list)
            elif args['Image order'] == "Oldest first":
                image_list.sort(key=lambda x: x[1].creation_date.timestamp())
            elif args['Image order'] == "Newest first":
                image_list.sort(key=lambda x: x[1].creation_date.timestamp(), reverse=True)
                
            if print_output:
                print('Making collage...')
            
            collage = makeCollage([i[2] for i in image_list],
                                  collage_type=args['Collage type'].lower(),
                                  max_recursion_depth=args['Max nesting depth'],
                                  spacing=args['Image spacing'],
                                  no_antialias=args['No intermediate antialiasing'],
                                  target_aspect_ratio=args['Collage aspect ratio'],
                                  max_collage_size=args['Collage max width/height'],
                                  round_image_corners_perc=args['Round image corners'],
                                  round_collage_corners_perc=args['Round collage corners'],
                                  rotate_images_max_deg=args['Rotate images'],
                                  show_recursion_depth=args['Show nesting level'])
            
            collage = resize_img_to_max_size(collage, args['Collage max width/height'])
            
            end_time = time.time()
            
            finished_str = f"Finished in {end_time - start_time:.2f} seconds"
            
            path='tmp.png'
            collage.save(path, format='PNG')
            if is_appex:
                console.quicklook(path)
            else:
                collage.show()
            
            # ask to save if user didn't already save from the quicklook
            last_asset = photos.get_assets()[-1]
            if ((datetime.datetime.now() - last_asset.creation_date).total_seconds() > 60 or last_asset.pixel_width != collage.width or last_asset.pixel_height != collage.height) and dialogs.alert('Save collage?', button1='Yes', button2='No', hide_cancel_button=True) == 1: 
                photos.create_image_asset(path)
                if print_output:
                    print('Collage saved to camera roll...')
                else:
                    try:
                        dialogs.alert('Collage saved to camera roll...', "", "OK", hide_cancel_button=True)
                    except KeyboardInterrupt:
                        pass
            os.remove(path)
            
            if print_output:
                print(finished_str)
            
            user_exit = dialogs.alert(title=finished_str, button1='New w/ new pics', button2='New w/ same pics', button3='Quit', hide_cancel_button=True)
            if user_exit == 1:
                image_assets = []
                PIL_images = []
                break
            
    if print_output:
        print('Exiting')


if __name__ == '__main__':
    main()
    pass
