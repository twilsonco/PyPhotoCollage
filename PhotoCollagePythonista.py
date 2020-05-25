'''
Created on May 24, 2020

@author: Tim Wilson
'''
import dialogs
import photos
import console

import datetime
import random
import math
import os
from PIL.ExifTags import TAGS
from PIL import ImageOps
from PIL import Image

# start partition problem algorithm from https://stackoverflow.com/a/7942946
# modified to act on list of images rather than the weights themselves

from operator import itemgetter

def linear_partition(seq, k, dataList = None):
    if k <= 0:
        return []
    n = len(seq) - 1
    if k > n:
        return map(lambda x: [x], seq)
    table, solution = linear_partition_table(seq, k)
    k, ans = k-2, []
    if dataList == None or len(dataList) != len(seq):
        while k >= 0:
            ans = [[seq[i] for i in range(solution[n-1][k]+1, n+1)]] + ans
            n, k = solution[n-1][k], k-1
        ans = [[seq[i] for i in range(0, n+1)]] + ans
    else:
        while k >= 0:
            ans = [[dataList[i] for i in range(solution[n-1][k]+1, n+1)]] + ans
            n, k = solution[n-1][k], k-1
        ans = [[dataList[i] for i in range(0, n+1)]] + ans
    return ans

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

def clamp(v,l,h):
    return l if v < l else h if v > h else v

# takes list of PIL image objects and returns the collage as a PIL image object
def makeCollage(imgList, spacing = 0, antialias = False, background=(0,0,0), aspectratiofactor = 1.0):
    # first downscale all images according to the minimum height of any image
    minHeight = min([img.height for img in imgList])
    totalWidth = sum([img.width for img in imgList])
    if antialias:
        imgList = [img.resize((int(img.width * img.height / minHeight),minHeight), Image.ANTIALIAS) if img.height > minHeight else img for img in imgList]
    else:
        imgList = [img.resize((int(img.width * img.height / minHeight),minHeight)) if img.height > minHeight else img for img in imgList]
    
    # generate the input for the partition problem algorithm
    # need list of aspect ratios and number of rows (partitions)
    imgHeights = [img.height for img in imgList]
    totalWidth = sum([img.height for img in imgList])
    avgWidth = totalWidth / len(imgList)
    targetWidth = avgWidth * math.sqrt(len(imgList) * aspectratiofactor)
    
    numRows = int(round(totalWidth / targetWidth))
    numRows = 2 if numRows < 2 else numRows
    aspectRatios = [int(img.width / img.height * 100) for img in imgList]
    
    # get nested list of images (each sublist is a row in the collage)
    imgRows = linear_partition(aspectRatios, numRows, imgList)
    
    # scale down larger rows to match the minimum row width
    rowWidths = [sum([img.width + spacing for img in row]) - spacing for row in imgRows]
    minRowWidth = min(rowWidths)
    rowWidthRatios = [minRowWidth / w for w in rowWidths]
    if antialias:
        imgRows = [[img.resize((int(img.width * widthRatio), int(img.height * widthRatio)), Image.ANTIALIAS) for img in row] for row,widthRatio in zip(imgRows, rowWidthRatios)]
    else:
        imgRows = [[img.resize((int(img.width * widthRatio), int(img.height * widthRatio))) for img in row] for row,widthRatio in zip(imgRows, rowWidthRatios)]
    
    # pupulate new image
    rowWidths = [sum([img.width + spacing for img in row]) - spacing for row in imgRows]
    rowHeights = [max([img.height for img in row]) for row in imgRows]
    minRowWidth = min(rowWidths)
    w,h = (minRowWidth, sum(rowHeights) + spacing * (numRows - 1))
    
    if background == (0,0,0):
        background += tuple([0])
    else:
        background += tuple([255])
    outImg = Image.new("RGBA", (w,h), background)
    xPos,yPos = (0,0)
    
    for row in imgRows:
        for img in row:
            outImg.paste(img, (xPos,yPos))
            xPos += img.width + spacing
            continue
        yPos += max([img.height for img in row]) + spacing
        xPos = 0
        continue
    
    return outImg

# modified (significantly) from https://github.com/delimitry/collage_maker
# this main function is for the CLI implementation
def main():
    console.clear()
    userExit = 0
    
    while userExit <= 1:
        # get images
        pilImages = []
        images = photos.pick_asset(title='Select images for collage', assets=photos.get_assets(), multi=True)
        if images is None:
            return
        
        # only keep photos
        images = [img for img in images if img.media_type == 'image']
        
        if len(images) < 3:
            userExit = dialogs.alert(title="Photo Collage", message="Select 3 or more images.", button1='OK', button3="Quit", hide_cancel_button=True)
        
        while userExit <= 2 and len(images) >= 3:
            # get user options
        #     while True:
            args = dialogs.form_dialog(title='Photo Collage', sections=[
                    ('Options', [
                        {'title':'Shuffle images:   ',
                         'key':'shuffle',
                         'type':'check',
                         'value':'True'},
                        {'title':'Spacing*:   ',
                         'key':'imagegap',
                         'type':'number',
                         'value':'0'},
                        {'title':'Collage width*:   ',
                         'key':'width',
                         'type':'number',
                         'value':'5000'},
                        {'title':'Collage height**:   ',
                         'key':'height',
                         'type':'number',
                         'value':'5000'}
                        ],
                        '*in pixels; **not used if width > 0'
                    ),
                    ('Advanced', [
                        {'title':'Initial image height*:   ',
                         'key':'initheight',
                         'type':'number',
                         'value':'500'},
                        {'title':'Scale collage aspect ratio**:   ',
                         'key':'aspectratiofactor',
                         'type':'number',
                         'value':'1.0'},
                        {'title':'Disable intermediate antialiasing***:   ',
                         'key':'noantialias',
                         'type':'check',
                         'value':'True'}],
                        '*lower values run faster and use less memory; **scaling mean image AR; ***final resize is always antialiased'
                    )])
                            
            try:
                args['imagegap'] = clamp(int(args['imagegap']), 0, 1000) if 'imagegap' in args else 0
                args['width'] = max([int(args['width']), 0]) if 'width' in args else 5000
                args['height'] = max([int(args['height']), 0]) if 'height' in args else 5000
                args['initheight'] = max([int(args['initheight']), 0]) if 'initheight' in args else 500
                args['aspectratiofactor'] = float(clamp(float(args['aspectratiofactor']), 0.05, 20.0)) if 'aspectratiofactor' in args else 1.0
                args['shuffle'] = True if args['shuffle'] == 'True' else False
                args['noantialias'] = True if args['noantialias'] == 'True' else False
            except:
                userExit = dialogs.alert(title='Photo Collage', message='There was one or more input errors. Check that your input values make sense and try again', button1='OK',button3='Quit',hide_cancel_button=True)
                continue
                
            # get PIL image objects for all the photos
            if userExit <= 1 or len(pilImages) != len(images):
                print('Loading photos...')
                pilImages = []
                for f in images:
                    img = f.get_image()
                    if args['initheight'] >= 50 and img.height > args['initheight']:
                        if args['noantialias']:
                            pilImages.append(img.resize((int(img.width / img.height * args['initheight']), args['initheight'])))
                        else:
                            pilImages.append(img.resize((int(img.width / img.height * args['initheight']), args['initheight']), Image.ANTIALIAS))
                    else:
                        pilImages.append(img)
            
            # shuffle images if needed
            if args['shuffle']:
                random.shuffle(pilImages)
                
            print('Making collage...')
            
            collage = makeCollage(pilImages, spacing=args['imagegap'], antialias=not args['noantialias'], aspectratiofactor=args['aspectratiofactor'])
            
            if args['width'] >= 50 and collage.width > args['width']:
                collage = collage.resize((args['width'], int(collage.height / collage.width * args['width'])), Image.ANTIALIAS)
                pass
            elif args['height'] >= 50 and collage.height > args['height']:
                collage = collage.resize((int(collage.width / collage.height * args['height']), args['height']), Image.ANTIALIAS)
                pass
            
            path='tmp.png'
            collage.save(path, format='PNG')
            console.quicklook(path)
            
            # ask to save if user didn't already save from the quicklook
            last_asset = photos.get_assets()[-1]
            if ((datetime.datetime.now() - last_asset.creation_date).total_seconds() > 60 or last_asset.pixel_width != collage.width or last_asset.pixel_height != collage.height) and dialogs.alert('Save collage?', button1='Yes', button2='No', hide_cancel_button=True) == 1: 
                photos.create_image_asset(path)
                print('Collage saved to camera roll...')
            os.remove(path)
            
            userExit = dialogs.alert(title="Finished!", button1='New w/ new pics', button2='New w/ same pics', button3='Quit', hide_cancel_button=True)
            
            print('Finished!')


if __name__ == '__main__':
    main()