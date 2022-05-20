from scipy import fftpack
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
import matplotlib.pyplot as plt

image = Image.open(r"D:\hala\New folder\fET3jT4-detective-conan-wallpaper.jpg")


def histogram():
    if image.mode == "RGB":
        r, g, b = image.split()
        rhistogramF = r.histogram()
        ghistogramF = g.histogram()
        bhistogramF = b.histogram()

        histogramindex = np.arange(256)
        plt.bar(x=histogramindex, height=rhistogramF)
        plt.show()
        plt.bar(x=histogramindex, height=ghistogramF)
        plt.show()
        plt.bar(x=histogramindex, height=bhistogramF)
        plt.show()
    else:
        image2 = ImageOps.grayscale(image)
        histogramF = image2.histogram()
        histogramindex = np.arange(256)
        plt.bar(x=histogramindex, height=histogramF)
        plt.show()


histogram()
Laplacian_kerl = [[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]]
box_kenl = [[1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9]]

sharp_kerl = [[0, -0.5, 0],
              [-0.5, 3, -0.5],
              [0, -0.5, 0]]


def convolutionFilter(filter, size):
    row = image.size[0]
    col = image.size[1]
    kerl = filter

    ksize = size
    offsite = ksize // 2
    input_pixels = image.load()

    finalImg = Image.new("RGB", (row, col))
    draw = ImageDraw.Draw(finalImg)

    for x in range(offsite, finalImg.width - offsite):
        for y in range(offsite, finalImg.height - offsite):
            acc = [0, 0, 0]
            for a in range(ksize):
                for b in range(ksize):
                    xn = x + a - offsite
                    yn = y + b - offsite
                    pixel = input_pixels[xn, yn]
                    acc[0] += pixel[0] * kerl[a][b]
                    acc[1] += pixel[1] * kerl[a][b]
                    acc[2] += pixel[2] * kerl[a][b]

            draw.point((x, y), (int(acc[0]), int(acc[1]), int(acc[2])))

    finalImg.save("final.png")


convolutionFilter(sharp_kerl, 3)


def imgEnhancement(value, mode):
   if (mode == "B"):
       bright = ImageEnhance.Brightness(image)
       output_image = bright.enhance(value)
       output_image.save("brightness.png")
   elif(mode == "D"):
       dark = ImageEnhance.Brightness(image)
       output_image = dark.enhance(1/value)
       output_image.save("darkness.png")


imgEnhancement(2,"B")
imgEnhancement(2,"D")

