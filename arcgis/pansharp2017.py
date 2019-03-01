# -*- coding: utf-8 -*-
import os
import arcpy
from arcpy import env
import time
from copy import deepcopy
import multiprocessing
env.workspace = 'D:/download/'


def orth(image):
    orthimage = os.path.splitext(image)[0] + "-orth" + ".tif"
    if os.path.exists(orthimage):
        print(orthimage + ' exists!')
        return orthimage
    try:
        # Ortho correct with Constant elevation
        arcpy.CreateOrthoCorrectedRasterDataset_management(
            image, orthimage, "CONSTANT_ELEVATION", "340", "#", "#", "#", "#")
    except:
        print("Create Ortho Corrected Raster Dataset example failed.")
        print(arcpy.GetMessages())
    return orthimage


def pansharp(rgbimage, panimage, savedir):
    pansharpiamge = os.path.join(savedir,
                                 os.path.split(rgbimage)[1][:-14] + ".tif")
    if os.path.exists(pansharpiamge):
        print(pansharpiamge + ' exits!')
        return None
    arcpy.CreatePansharpenedRasterDataset_management(
        rgbimage, "1", "2", "3", "4", pansharpiamge, panimage, "Brovey")


def processsingle(imagefolder, savedir):
    datadir, imagedir = os.path.split(imagefolder)
    mssorth = None
    panorth = None
    images = os.listdir(imagefolder)
    for image in images:
        fullpath = os.path.join(datadir, imagedir, image)
        if image[-5:] == ".tiff":
            if image[-9:-6] == "MSS":
                mssorth = orth(fullpath)
                print(imagedir + " mss-orth")
            if image[-9:-6] == "PAN":
                panorth = orth(fullpath)
                print(imagedir + " pan-orth")
    if mssorth and panorth:
        cur_savedir = os.path.join(savedir, imagedir)
        if not os.path.exists(cur_savedir):
            os.makedirs(cur_savedir)
        pansharp(mssorth, panorth, cur_savedir)
        print("pansharpiamge")
    else:
        print(imagedir + "errer")


def main():
    datadir = r'D:\2017-2018\2017'
    allimagedirs = os.listdir(datadir)
    logfile = open(r"D:\Code\HTHT\python\arcgis\log.txt", 'a')
    savedir = r'D:\fusion\2017'
    num = 1
    pronum = 2
    pro_pool = multiprocessing.Pool(pronum)
    for imagedir in allimagedirs:
        cur_folder = os.path.join(datadir, imagedir)
        if os.path.isfile(cur_folder):
            continue
        # kwargs = {"imagefolder": cur_folder, "savedir": savedir}
        # result = pro_pool.apply_async(
        #     func=processsingle, args="", kwds=deepcopy(kwargs))
        processsingle(cur_folder, savedir)
        num += 1
    logfile.close()
    print('Finish!')

if __name__ == "__main__":
    main()