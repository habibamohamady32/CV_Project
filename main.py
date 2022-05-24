import cv2
from scipy import fftpack
import numpy as np
from PIL import Image, ImageDraw , ImageOps,ImageEnhance
import matplotlib.pyplot as plt
from Kmeans_task import *
##-----------------------------K-means----------------------------------##
k= 3
img = cv2.imread("Image_to_be_segmented.jpg")
cv2.imwrite(f"images/SegmantedImage.jpg", reducedColors_image(img,k))
##-----------------------------K-means----------------------------------##

#---------------------------BandReject------------------------------#
inputImage = Image.open('input.jpg')
inputImage = ImageOps.grayscale(inputImage)

#convert image to numpy array
inputImage_np = np.array(inputImage)

#fft of image
inputImage_fft = fftpack.fftshift(fftpack.fft2(inputImage))

#Create a band reject filter image
x , y = inputImage_np.shape[0] , inputImage_np.shape[1]

#size of circles
c1_x , c1_y = 100 , 100
c2_x , c2_y = 45 , 45
#create a 2 boxes
box1 = ((x/2)-(c1_x/2),(y/2)-(c1_y/2),(x/2)+(c1_x/2),(y/2)+(c1_y/2))
box2 = ((x/2)-(c2_x/2),(y/2)-(c2_y/2),(x/2)+(c2_x/2),(y/2)+(c2_y/2))
print(box1)
print(box2)
band_pass = Image.new("L" , (inputImage_np.shape[0] , inputImage_np.shape[1]) , color=1)
print(np.shape(band_pass))
drawF = ImageDraw.Draw(band_pass)
drawF.ellipse(box1, fill=0)
drawF.ellipse(box2, fill=1)
band_pass_np = np.array(band_pass)

plt.imshow(band_pass)
plt.show()

#multiply inputImage & Filter
filterdImage = np.multiply(inputImage_fft , band_pass_np)

#inverse fft
ifft = fftpack.ifft2(fftpack.ifftshift(filterdImage))
finalImg = Image.new("L" , (inputImage_np.shape[0] , inputImage_np.shape[1]))
draw = ImageDraw.Draw(finalImg)

for x in range(inputImage_np.shape[0]):
    for y in range(inputImage_np.shape[1]):
        draw.point((x, y), (int(ifft[y][x])))

#save the image
finalImg.save("images/Output.jpg")
#---------------------------BandReject------------------------------#

#---------------------------Equalization------------------------------#
imge = Image.open("Image_before_equalization.jpg")

imge = ImageOps.grayscale(imge)

def gray_Eq(image=None):

    imge2 = ImageOps.grayscale(imge)
    imgArray = np.array(imge2)

    L = pow(2, 8)
    n = imgArray.shape[0] * imgArray.shape[1]
    Ln = L / n

    Gray = {}

    for i in range(imgArray.shape[0]):
        for j in range(imgArray.shape[1]):
            if (imgArray[i][j] not in Gray):
                Gray[imgArray[i][j]] = []
                Gray[imgArray[i][j]].append(1)
            elif (imgArray[i][j] in Gray):
                Gray[imgArray[i][j]][0] += 1

    Items = Gray.items()
    Sort = sorted(Items)
    DSort = {}
    print(Gray)
    print(Sort)
    T = 0
    for key in Sort:
        DSort[key[0]] = key[1]
        T += DSort[key[0]][0]
        eq = (Ln * T) - 1
        DSort[key[0]].append(T)
        DSort[key[0]].append(eq)

    print(DSort)

    finalImg = Image.new("L", (imgArray.shape[1], imgArray.shape[0]))
    draw = ImageDraw.Draw(finalImg)

    for x in range(imgArray.shape[0]):
        for y in range(imgArray.shape[1]):
            draw.point((y, x), int(DSort[imgArray[x][y]][2]))

    finalImg.save("images/final.png")
    histogramF = image.histogram()
    histogramindex = np.arange(256)

    plt.bar(x=histogramindex, height=histogramF)
    plt.show()

    histogramF = finalImg.histogram()
    plt.bar(x=histogramindex, height=histogramF)
    plt.show()

def RGB_Eq():
    r, g, b = imge.split()
    L = pow(2, 8)

    imgArray = np.array(imge)

    n = imgArray.shape[0] * imgArray.shape[1]

    Ln = L / n

    rArray = np.array(r)
    gArray = np.array(g)
    bArray = np.array(b)

    rGray = {}
    gGray = {}
    bGray = {}

    for i in range(imgArray.shape[1]):
        for j in range(imgArray.shape[0]):
            if (rArray[i][j] not in rGray):
                rGray[rArray[i][j]] = []
                rGray[rArray[i][j]].append(1)
            elif (rArray[i][j] in rGray):
                rGray[rArray[i][j]][0] += 1

            if (gArray[i][j] not in gGray):
                gGray[gArray[i][j]] = []
                gGray[gArray[i][j]].append(1)
            elif (gArray[i][j] in gGray):
                gGray[gArray[i][j]][0] += 1

            if (bArray[i][j] not in bGray):
                bGray[bArray[i][j]] = []
                bGray[bArray[i][j]].append(1)
            elif (bArray[i][j] in bGray):
                bGray[bArray[i][j]][0] += 1

    rItems = rGray.items()
    gItems = gGray.items()
    bItems = bGray.items()

    rSort = sorted(rItems)
    gSort = sorted(gItems)
    bSort = sorted(bItems)

    rDSort = {}
    gDSort = {}
    bDSort = {}

    T = 0
    for key in rSort:
        rDSort[key[0]] = key[1]
        T += rDSort[key[0]][0]
        eq = (Ln * T) - 1
        rDSort[key[0]].append(T)
        rDSort[key[0]].append(eq)

    T = 0
    for key in gSort:
        gDSort[key[0]] = key[1]
        T += gDSort[key[0]][0]
        eq = (Ln * T) - 1
        gDSort[key[0]].append(T)
        gDSort[key[0]].append(eq)

    T = 0
    for key in bSort:
        bDSort[key[0]] = key[1]
        T += bDSort[key[0]][0]
        eq = (Ln * T) - 1
        bDSort[key[0]].append(T)
        bDSort[key[0]].append(eq)

    finalImg = Image.new("RGB", (imgArray.shape[1], imgArray.shape[0]))
    draw = ImageDraw.Draw(finalImg)

    for x in range(imgArray.shape[1]):
        for y in range(imgArray.shape[0]):
            red = rDSort[rArray[x][y]][2]
            green = gDSort[gArray[x][y]][2]
            blue = bDSort[bArray[x][y]][2]
            draw.point((y, x), (int(red), int(green), int(blue)))

    finalImg.save("images/Equalized_Image.png")


if imge.mode == "RGB":
    RGB_Eq()
else:
    gray_Eq(imge)
#---------------------------Equalization------------------------------#

#-----------------------------Histogram----------------------------------#

image = Image.open("spidy.jpg")


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

    finalImg.save("images/filteredImg.png")


convolutionFilter(sharp_kerl, 3)


def imgEnhancement(value, mode):
   if (mode == "B"):
       bright = ImageEnhance.Brightness(image)
       output_image = bright.enhance(value)
       output_image.save("images/brightness.png")
   elif(mode == "D"):
       dark = ImageEnhance.Brightness(image)
       output_image = dark.enhance(1/value)
       output_image.save("images/darkness.png")


imgEnhancement(2,"B")
imgEnhancement(2,"D")
#-----------------------------Histogram----------------------------------#