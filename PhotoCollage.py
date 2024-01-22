'''
Created on May 24, 2020

@author: Tim Wilson
'''
import re
import argparse
import os
import random
import math
from PIL.ExifTags import TAGS
from PIL import Image, ImageDraw, ImageChops, ImageOps
from operator import itemgetter

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

def add_corners(im, rad, supersample=3):
    # Create a larger image for supersampling
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


def clamp(v,l,h):
    return l if v < l else h if v > h else v

# takes list of PIL image objects and returns the collage as a PIL image object
# collage_type can be one of "nested", "row", or "column"
def makeCollage(img_list,
                collage_type = "nested", 
                recursion_depth = 0, 
                max_recursion_depth = 2,
                min_batch_size_in = 3,
                max_batch_size_in = 10,
                spacing = 0, 
                no_antialias = False, 
                background=(0,0,0), 
                target_aspect_ratio = 1.0, 
                max_collage_size = 0, 
                round_image_corners_perc = 0.0,
                round_collage_corners_perc = 0.0,
                show_recursion_depth = False):
    # check img_ordering and collage_type args
    if collage_type not in ["nested", "row", "column"]:
        raise ValueError("collage_type must be one of 'nested', 'by_row', or 'by_column'")
    
    if collage_type == "nested":
        max_recursion_depth = max(max_recursion_depth, 1)

    # perform processing of images for top-level call only
    if recursion_depth == 0:
        # round corners of images as percentage of shortest edge if requested
        if round_image_corners_perc > 0.0:
            img_list = [add_corners(img, int(min(img.width, img.height) * 0.5 * round_image_corners_perc / 100.0)) for img in img_list]
        # resize all images so that the longest edge is less than or equal to max_collage_size (if max_collage_size is greater than 0)
        if max_collage_size > 0:
            tmp_img_list = []
            for img in img_list:
                if img.width > max_collage_size:
                    if no_antialias:
                        tmp_img_list.append(img.resize((max_collage_size, int(img.height / img.width * max_collage_size))))
                    else:
                        tmp_img_list.append(img.resize((max_collage_size, int(img.height / img.width * max_collage_size)), Image.LANCZOS))
                elif img.height > max_collage_size:
                    if no_antialias:
                        tmp_img_list.append(img.resize((int(img.width / img.height * max_collage_size), max_collage_size)))
                    else:
                        tmp_img_list.append(img.resize((int(img.width / img.height * max_collage_size), max_collage_size), Image.LANCZOS))
                else:
                    tmp_img_list.append(img)
            img_list = tmp_img_list
            del tmp_img_list
    
    # for column-major collage, set the do_rotate flag to True. For nested collage, set to false for top-level call and randint(0,1) for recursive calls
    if collage_type == "column":
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
        elif target_aspect_ratio <= ar_if_vertical:
            do_recurse = False
        elif recursion_depth > max_recursion_depth:
            do_recurse = False
        elif max_num_batches <= 3:
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
                    batch_img_list = []
                    for img_num in batch:
                        img = img_list[img_num]
                        if no_antialias:
                            batch_img_list.append(img.resize((int(img.width * downscale_factor), int(img.height * downscale_factor))))
                        else:
                            batch_img_list.append(img.resize((int(img.width * downscale_factor), int(img.height * downscale_factor)), Image.LANCZOS))
                    # aspect ratio of batch is a random in [0.5, 2]
                    batch_target_aspect_ratio = random.uniform(0.5, 2.0)
                    batch_max_collage_size = int(max_collage_size * downscale_factor)
                    img = makeCollage(batch_img_list, 
                                                    collage_type=collage_type,
                                                    recursion_depth=recursion_depth + 1, 
                                                    max_recursion_depth=max_recursion_depth, 
                                                    min_batch_size_in=min_batch_size_in, 
                                                    max_batch_size_in=max_batch_size_in, 
                                                    spacing=int(spacing*1.2),
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
        border_width_factor = 0.03 * (recursion_depth + 1)
        border_color = border_colors[recursion_depth % len(border_colors)]
        img_list = [ImageOps.expand(img, border=int((img.width + img.height) / 2 * border_width_factor), fill=border_color) for img in img_list]
        
    # rotate images if needed
    if do_rotate:
        img_list = [img.transpose(Image.ROTATE_90) for img in img_list]
    
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
        if no_antialias:
            img_list = [img.resize((int(img.width * common_height / img.height), common_height)) for img in img_list]
        else:
            img_list = [img.resize((int(img.width * common_height / img.height), common_height), Image.LANCZOS) for img in img_list]
    else:
        # resize to common_width
        if no_antialias:
            img_list = [img.resize((common_width, int(img.height * common_width / img.width))) for img in img_list]
        else:
            img_list = [img.resize((common_width, int(img.height * common_width / img.width)), Image.LANCZOS) for img in img_list]
    
    if num_rows == 1:
        img_rows = [img_list]
    elif num_rows == num_images:
        img_rows = [[img] for img in img_list]
    else:
        aspect_ratios = [int(img.width / img.height * 100) for img in img_list]
    
        # get nested list of images (each sublist is a row in the collage)
        img_rows = linear_partition(aspect_ratios, num_rows, img_list, do_rotate)
    
        # # scale down larger rows to match the minimum row width
        # row_widths = [sum([img.width + spacing for img in row]) - spacing for row in img_rows]
        # min_row_width = math.ceil(min(row_widths))
        # if no_antialias:
        #     img_rows = [[img.resize((int(img.width * min_row_width / row_widths[i]), int(img.height * min_row_width / row_widths[i]))) for img in row] for i,row in enumerate(img_rows)]
        # else:
        #     img_rows = [[img.resize((int(img.width * min_row_width / row_widths[i]), int(img.height * min_row_width / row_widths[i])), Image.LANCZOS) for img in row] for i,row in enumerate(img_rows)]
    
    
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
    if no_antialias:
        row_imgs = [img.resize((min_row_width, int(img.height * min_row_width / img.width))) for img in row_imgs]
    else:
        row_imgs = [img.resize((min_row_width, int(img.height * min_row_width / img.width)), Image.LANCZOS) for img in row_imgs]
    
    # pupulate new image
    row_widths = [img.width for img in row_imgs]
    row_heights = [img.height for img in row_imgs]
    w, h = (row_widths[0], sum(row_heights) + spacing * (num_rows - 1))
    
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
        rad = int(min(out_img.width, out_img.height) * 0.5 * round_collage_corners_perc / 100.0)
        out_img = add_corners(out_img, rad)
    
    return out_img

def make_arg_parser():
    def rgb(s):
        try:
            rgb = (0 if v < 0 else 255 if v > 255 else v for v in map(int, s.split(',')))
            return rgb
        except:
            raise argparse.ArgumentTypeError('Background must be (r,g,b) --> "(0,0,0)" to "(255,255,255)"')
    parse = argparse.ArgumentParser(description='Photo collage maker')
    parse.add_argument('-f', '--folder', dest='folder', help='folder with 3 or more images (*.jpg, *.jpeg, *.png)', default=False)
    parse.add_argument('-F', '--file', dest='file', help='file with newline separated list of 3 or more files', default=False)
    parse.add_argument('-o', '--output', dest='output', help='output collage image filename', default='collage.png')
    parse.add_argument('-t', '--collage-type', dest='collage_type', help='collage type (default: nested; possible: row, column)', default='nested')
    parse.add_argument('-O', '--order', dest='order', help='order of images (default: input_order; possible: filename, shuffled, oldest_first, newest_first, input)', default='input_order')
    parse.add_argument('-S', '--max_collage_size', dest='max_size', type=int, help='cap the longest edge (width or height) of resulting collage', default=5000)
    parse.add_argument('-r', '--target-aspect-ratio', dest='target_aspect_ratio', type=float, help='target aspect ratio for collage', default=1.0)
    parse.add_argument('-g', '--gap-between-images', dest='imagegap', type=int, help='number of pixels of transparent space (if saving as png file; otherwise black or specified background color) to add between neighboring images', default=0)
    parse.add_argument('-m', '--round-image-corners-perc', dest='round_image_corners_perc', type=float, help='percentage of shortest image edge to use as radius for rounding image corners (0.0 to 100.0)', default=0.0)
    parse.add_argument('-M', '--round-collage-corners-perc', dest='round_collage_corners_perc', type=float, help='percentage of shortest collage edge to use as radius for rounding collage corners (0.0 to 100.0)', default=0.0)
    parse.add_argument('-b', '--background-color', dest='background', type=rgb, help='color (r,g,b) to use for background if spacing is added between images', default=(0,0,0))
    parse.add_argument('-c', '--count', dest='count', type=int, help='count of images to use, if fewer are desired than those specified or contained in the specified folder', default=0)
    parse.add_argument('-a', '--no-no_antialias-when-resizing', dest='noantialias', action='store_false', help='for performance, disable antialiasing on intermediate resizing of images (runs faster but output image looks worse; final resize is always antialiased)')
    parse.add_argument('-i', '--init-height', dest='init_height', type=int, help='to derease necessary memory, resize images on input to set height', default=1000)
    parse.add_argument('-R', '--max-recursion-depth', dest='max_recursion_depth', type=int, help='maximum number of levels of nesting. The more levels there are, the larger the size difference will be between the smallest and largest images in the resulting collage (default: 2)', default=2)
    parse.add_argument('-s', '--show-recursion-depth', dest='show_recursion_depth', action='store_true', help='show recursion depth by adding border to images')
    parse.add_argument('files', nargs='*')
    
    return parse

# modified (significantly) from https://github.com/delimitry/collage_maker
# this main function is for the CLI implementation
def main(args_in=None):
    parse = make_arg_parser()
    
    if args_in:
        args = parse.parse_args(args_in)
    else:
        args = parse.parse_args()
    if not args.file and not args.folder and not args.files:
        parse.print_help()
        exit(1)

    # get images
    images = args.files
    if args.folder:
        files = [os.path.join(args.folder, fn) for fn in os.listdir(args.folder)]
        images = [fn for fn in files if re.findall("jpg|png|jpeg", fn.lower().split(".")[-1])]
    
    if args.file:
        images = []
        with open(args.file, 'r') as f:
            for line in f:
                images.append(line.strip())
    # elif args.folder:
    #     images = []
    #     for root, _, files in os.walk(args.folder):
    #         for name in files:
    #             if re.findall("jpg|png|jpeg", name.lower().split(".")[-1]):
    #                 fname = os.path.join(root, name)
    #                 images.append(fname)
                    
    print(f'Found {len(images)} images')
    
    if len(images) < 3:
        print("Need to use 3 or more images. Try again")
        return

    # apply image ordering
    if args.order == "shuffled":
        random.shuffle(images)
    elif args.order == "filename":
        images.sort()
    elif args.order == "oldest_first":
        images.sort(key=lambda x: os.path.getmtime(x))
    elif args.order == "newest_first":
        images.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    if args.count > 2:
        images = images[:args.count]
        
    print(f'Using {len(images)} images')
        
    # get PIL image objects for all the photos
    print('Loading photos...')
    pil_images = []
    for f in images:
        img = Image.open(f)
        # Need to explicitly tell PIL to rotate image if EXIF orientation data is present
        exif = img.getexif()
        # Remove all exif tags
        for k in exif.keys():
            if k != 0x0112:
                exif[k] = None # If I don't set it to None first (or print it) the del fails for some reason. 
                del exif[k]
        # Put the new exif object in the original image
        new_exif = exif.tobytes()
        img.info["exif"] = new_exif
        # Rotate the image
        img = ImageOps.exif_transpose(img)
        if args.init_height > 2 and img.height > args.init_height:
            if args.noantialias:
                pil_images.append(img.resize((int(img.width / img.height * args.init_height),args.init_height)))
            else:
                pil_images.append(img.resize((int(img.width / img.height * args.init_height),args.init_height), Image.LANCZOS))
        else:
            pil_images.append(img)

        
    print('Making collage...')
    
    collage = makeCollage(pil_images, 
                          collage_type=args.collage_type, 
                          max_recursion_depth=args.max_recursion_depth,
                          min_batch_size_in=3, 
                          max_batch_size_in=10, 
                          spacing=args.imagegap, 
                          no_antialias=args.noantialias, 
                          background=args.background, 
                          target_aspect_ratio=args.target_aspect_ratio, 
                          max_collage_size=args.max_size, 
                          round_image_corners_perc=args.round_image_corners_perc, 
                          round_collage_corners_perc=args.round_collage_corners_perc,
                          show_recursion_depth=args.show_recursion_depth)
    
    if args.max_size > 0 and collage.width > args.max_size:
        collage = collage.resize((args.max_size, int(collage.height / collage.width * args.max_size)), Image.LANCZOS)
        pass
    elif args.max_size > 0 and collage.height > args.max_size:
        collage = collage.resize((int(collage.width / collage.height * args.max_size), args.max_size), Image.LANCZOS)
        pass
    
    output = args.output
    if output == 'collage.png':
        output = images[0] + '.collage.png'
    collage.save(output)
    
    print(f'Collage is ready at {output}!')


if __name__ == '__main__':
    # test args array using input folder at /Users/haiiro/Downloads/collage_in_test
    args = ['-f', '/Users/haiiro/Downloads/collage_in_test', '-S', '5000', '-O', 'oldest_first', '-t', 'nested', '-R', '1', '-g', '10', '-r', '1.0', '-a', '-o', '/Users/haiiro/Downloads/collage_test.png', '-m', '20', '-c', '60']
    
    main(args)  
