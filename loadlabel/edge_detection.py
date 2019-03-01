import cv2
import numpy as np
from matplotlib import pyplot as plt


def structedge_orient(img, model):
    img = img / 255
    img = img.astype(np.float32)
    # edges = np.zeros(img.shape, np.float32)
    # orientation_map = np.zeros((img.shape[0], img.shape[1], 1), np.float32)
    pDollar = cv2.ximgproc.createStructuredEdgeDetection(model)
    edges = pDollar.detectEdges(img)
    orientation_map = pDollar.computeOrientation(edges)
    return edges, orientation_map


def main():
    # imagename = r'D:\data\dataset\Stanford_background\iccv09Data\images\0000047.jpg'
    # imagename = r'D:\data\dataset\14.jpeg'
    imagename = "D:/data/dataset/0000382.jpg"
    img = cv2.imread(imagename)
    img = img[:, :, ::-1]
    plt.subplot(221)
    plt.imshow(img)

    # 获取边缘
    model = r'D:/Code/HTHT/python/loadlabel/model.yml'
    result = structedge_orient(img, model)
    edge = (result[0] * 255).astype(np.uint8)
    plt.subplot(222)
    plt.imshow(edge)
    # 阈值分割
    ad_threshold = cv2.adaptiveThreshold(
        edge, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0.5)
    # _, threshold = cv2.threshold(edge, 200, 255,
    #                              cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV )
    plt.imshow(ad_threshold)
    # 查找轮廓
    contours, hierarchy = cv2.findContours(
        ad_threshold, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    markers = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, [(i + 3) * 2, (i + 3) * 2,
                                                (i + 3) * 2])
    plt.subplot(223)
    plt.imshow((markers / markers.max() * 255).astype(np.uint8))
    # 分水岭变换
    markers3 = cv2.watershed(img, markers)
    img[markers3 == -1] = [255, 0, 0]
    plt.subplot(224)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()
