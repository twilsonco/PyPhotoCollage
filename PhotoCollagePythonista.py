'''
Created on May 24, 2020

@author: Tim Wilson twilsonco @t gmail d.t com
'''
import dialogs
import photos
import console

import datetime
import random
import math
import os
from PIL import ImageOps
from PIL import Image

defaultArgs = {
    'imagegap':'0',
    'width':'5000',
    'height':'5000',
    'initheight':'500',
    'targetaspectratio':'1.0',
    'shuffle':'True',
    'noantialias':'True'
}

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
def makeCollage(imgList, spacing = 0, antialias = False, background=(0,0,0), targetaspectratio = 1.0):
    # first downscale all images according to the minimum height of any image
#     minHeight = min([img.height for img in imgList])
#     if antialias:
#         imgList = [img.resize((int(img.width / img.height * minHeight),minHeight), Image.ANTIALIAS) if img.height > minHeight else img for img in imgList]
#     else:
#         imgList = [img.resize((int(img.width / img.height * minHeight),minHeight)) if img.height > minHeight else img for img in imgList]
        
    # first upscale all images according to the maximum height of any image
    maxHeight = max([img.height for img in imgList])
    if antialias:
        imgList = [img.resize((int(img.width / img.height * maxHeight),maxHeight), Image.ANTIALIAS) if img.height < maxHeight else img for img in imgList]
    else:
        imgList = [img.resize((int(img.width / img.height * maxHeight),maxHeight)) if img.height < maxHeight else img for img in imgList]
        
    # generate the input for the partition problem algorithm
    # need list of aspect ratios and number of rows (partitions)
    imgHeights = [img.height for img in imgList]
    totalWidth = sum([img.width for img in imgList])
    totalHeight = sum([img.height for img in imgList])
    avgWidth = totalWidth / len(imgList)
    avgHeight = totalHeight / len(imgList)
    targetWidth = (avgHeight + avgWidth) / 2 * math.sqrt(len(imgList) * targetaspectratio)
    
    numRows = clamp(int(round(totalWidth / targetWidth)), 1, len(imgList))
    if numRows == 1:
        imgRows = [imgList]
    elif numRows == len(imgList):
        imgRows = [[img] for img in imgList]
    else:
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
    
def getArgs(args = None):
    import copy
    
    if args is None:
        args = copy.deepcopy(defaultArgs)
    else:
        args['imagegap'] = str(clamp(int(args['imagegap']), 0, 1000) if 'imagegap' in args else 0)
        args['width'] = str(max([int(args['width']), 0]) if 'width' in args else 5000)
        args['height'] = str(max([int(args['height']), 0]) if 'height' in args else 5000)
        args['initheight'] = str(max([int(args['initheight']), 0]) if 'initheight' in args else 500)
        args['targetaspectratio'] = str(float(clamp(float(args['targetaspectratio']), 0.001, 1000.0)) if 'targetaspectratio' in args else 1.0)
        args['shuffle'] = str(True if args['shuffle'] == 'True' else False)
        args['noantialias'] = str(True if args['noantialias'] == 'True' else False)
        
    args = dialogs.form_dialog(title='Photo Collage', sections=[
                ('Options', [
                    {'title':'Shuffle images:   ',
                     'key':'shuffle',
                     'type':'check',
                     'value':args['shuffle']},
                    {'title':'Spacing*:   ',
                     'key':'imagegap',
                     'type':'number',
                     'value':args['imagegap']},
                    {'title':'Collage width*:   ',
                     'key':'width',
                     'type':'number',
                     'value':args['width']},
                    {'title':'Collage height**:   ',
                     'key':'height',
                     'type':'number',
                     'value':args['height']}
                    ],
                    '*in pixels; **not used if width > 0'
                ),
                ('Advanced', [
                    {'title':'Initial image height*:   ',
                     'key':'initheight',
                     'type':'number',
                     'value':args['initheight']},
                    {'title':'Collage aspect ratio:   ',
                     'key':'targetaspectratio',
                     'type':'number',
                     'value':args['targetaspectratio']},
    #                 {'title':'Background color***:   ',
    #                  'key':'background',
    #                  'type':'text',
    #                  'value':'0,0,0'},
                    {'title':'Disable intermediate antialiasing**:   ',
                     'key':'noantialias',
                     'type':'check',
                     'value':args['noantialias']}],
                    '*lower values run faster and use less memory; **final resize is always antialiased'
                )])
                
    try:
        args['imagegap'] = clamp(int(args['imagegap']), 0, 1000) if 'imagegap' in args else 0
        args['width'] = max([int(args['width']), 0]) if 'width' in args else 5000
        args['height'] = max([int(args['height']), 0]) if 'height' in args else 5000
        args['initheight'] = max([int(args['initheight']), 0]) if 'initheight' in args else 500
        args['targetaspectratio'] = float(clamp(float(args['targetaspectratio']), 0.05, 20.0)) if 'targetaspectratio' in args else 1.0
        args['shuffle'] = True if args['shuffle'] == 'True' or args['shuffle'] else False
        args['noantialias'] = True if args['noantialias'] == 'True' or args['noantialias'] else False
        userExit = 2
    except:
        userExit = dialogs.alert(title='Photo Collage', message='There was one or more input errors. Check that your input values make sense and try again', button1='OK',button3='Quit',hide_cancel_button=True)
    
    return args,userExit

# modified (significantly) from https://github.com/delimitry/collage_maker
# this main function is for the CLI implementation
def main():
    console.clear()
    userExit = 0
    args = None
    imageAssets = []
    pilImages = []
    
    while userExit <= 1:
        # get images
        if len(imageAssets) < 2:
            pilImages = []
            imageAssets = photos.pick_asset(title='Select images for collage', assets=photos.get_assets(), multi=True)
        if imageAssets is None:
            return
        
        # only keep photos
        imageAssets = [img for img in imageAssets if img.media_type == 'image']
        
        if len(imageAssets) < 2:
            userExit = dialogs.alert(title="Photo Collage", message="Select 3 or more images.", button1='OK', button3="Quit", hide_cancel_button=True)
        
        while userExit <= 2 and len(imageAssets) >= 2:
            # get user options
            args,userExit = getArgs(args)
                
            # get PIL image objects for all the photos
            if userExit <= 1 or len(pilImages) != len(imageAssets):
                print('Loading photos...')
                pilImages = []
                for f in imageAssets:
                    img = f.get_image()
                    if args['initheight'] >= 50 and img.height > args['initheight']:
                        if args['noantialias']:
                            pilImages.append(img.resize((int(img.width / img.height * args['initheight']), args['initheight'])))
                        else:
                            pilImages.append(img.resize((int(img.width / img.height * args['initheight']), args['initheight']), Image.ANTIALIAS))
                    else:
                        pilImages.append(img)
                imageList = [(i,imageAssets[i],pilImages[i]) for i in range(len(pilImages))]
                
            
            # shuffle images if needed
            if args['shuffle']:
                random.shuffle(imageList)
            else:
                imageList.sort(key=lambda img:img[0])
                
            print('Making collage...')
            
            collage = makeCollage([i[2] for i in imageList], spacing=args['imagegap'], antialias=not args['noantialias'], targetaspectratio=args['targetaspectratio'])
            
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
            if userExit == 1:
                imageAssets = []
            
            print('Finished!')


if __name__ == '__main__':
    main()
    pass
