from PIL import Image
import numpy as np

colorlist = [[255, 228, 196], [0, 197, 205], [102, 205, 170], [112, 128, 144],
             [123, 104, 238], [78, 238, 148], [154, 192, 205], [238, 174, 238]]


def getimageinfo(info):
    name = info[:7]
    size = (info[8:11], info[12:15])
    location = info[16:]
    return name, size, location


def loadtxtimg(txtdir, imginfo):
    if not imginfo[0]:
        return False
    imgnp = np.zeros(imginfo[1])
    txtfn = open(txtdir + imginfo[0], 'r')
    for i in range(imginfo[1][0]):
        row = txtfn.read(2 * imginfo[1][1])
        for j in range(imginfo[1][1]):
            imgnp[i, j]


# def convert_txt_img(txtfile,size):


def main():
    txtfile = r'D:\data\dataset\Stanford_background\iccv09Data\horizons.txt'
    txtfn = open(txtfile, 'r')
    info = txtfn.read(26)
    while info:
        imginfo = getimageinfo(info)
        info = txtfn.read(26)


main()