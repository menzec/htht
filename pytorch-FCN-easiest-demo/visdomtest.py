import visdom
import os
import cv2
import time


def main():
    vis = visdom.Visdom()
    vis.text("train detail")
    datadir = r"D:\Code\HTHT\python\pytorch-FCN-easiest-demo"
    for imgname in os.listdir(os.path.join(datadir, "bag_data")):
        img = cv2.imread(os.path.join(datadir, "bag_data", imgname))
        img = img.transpose(2, 0, 1)
        imglabel = cv2.imread(os.path.join(datadir, "bag_data_mask", imgname), 0)
        vis.image(img, win='img', opts=dict(title='natral image'))
        vis.image(imglabel, win='img_label', opts=dict(title='label'))
        time.sleep(2)


if __name__ == "__main__":
    main()