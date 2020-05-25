# PyPhotoCollage
Combines 3 or more images into a collage, maintaining image aspect ratios and distributing images evenly over collage rows.

Dissatisfied with the iOS Shortcuts image gridding capability, I wanted a photo collage tool that can distribute images evenly amoung rows while maintain image aspect ratio.

After looking around, the idea for this tool came from [Johannes Treitz's](https://medium.com/@jtreitz) [blog post](https://medium.com/@jtreitz/the-algorithm-for-a-perfectly-balanced-photo-gallery-914c94a5d8af).
The problem is reduced to the [partition problem](http://www8.cs.umu.se/kurser/TDBAfl/VT06/algorithms/BOOK/BOOK2/NODE45.HTM), for which I used [this SO implementation](https://stackoverflow.com/a/7942946).

Options for the CLI version:
```usage: PhotoCollage.py [-h] [-f FOLDER] [-F FILE] [-o OUTPUT] [-W WIDTH]
                       [-H HEIGHT] [-i INITHEIGHT] [-s] [-g IMAGEGAP]
                       [-b BACKGROUND] [-c COUNT] [-r ASPECTRATIOFACTOR] [-a]

Photo collage maker

optional arguments:
  -h, --help            show this help message and exit
  -f FOLDER, --folder FOLDER
                        folder with images (*.jpg, *.jpeg, *.png)
  -F FILE, --file FILE  file with newline separated list of files
  -o OUTPUT, --output OUTPUT
                        output collage image filename
  -W WIDTH, --width WIDTH
                        resulting collage image height (mutually exclusive
                        with --height)
  -H HEIGHT, --height HEIGHT
                        resulting collage image height (mutually exclusive
                        with --width)
  -i INITHEIGHT, --initheight INITHEIGHT
                        resize images on input to set height
  -s, --shuffle         enable images shuffle
  -g IMAGEGAP, --gap-between-images IMAGEGAP
                        number of pixels of transparent space (if saving as
                        png file; otherwise black or specified background
                        color) to add between neighboring images
  -b BACKGROUND, --background-color BACKGROUND
                        color (r,g,b) to use for background if spacing is
                        added between images
  -c COUNT, --count COUNT
                        count of images to use
  -r ASPECTRATIOFACTOR, --scale-aspect-ratio ASPECTRATIOFACTOR
                        aspect ratio scaling factor, multiplied by the average
                        aspect ratio of the input images to determine the
                        output aspect ratio
  -a, --no-antialias-when-resizing
                        disable antialiasing on intermediate resizing of
                        images (runs faster but output image looks worse;
                        final resize is always antialiased)```
