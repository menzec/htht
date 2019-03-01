import cv2
import matplotlib.pyplot as plt
import numpy as np
from queue import PriorityQueue
from PIL import Image

def main():
    img = np.zeros((400, 400), np.uint8)
    cv2.circle(img, (150, 150), 100, 255, -1)
    cv2.circle(img, (250, 250), 100, 255, -1)
    cv2.imshow("circle", img)


    # 进行了距离变换
    dist = cv2.distanceTransform(img, cv2.DIST_L2,
                                 cv2.DIST_MASK_3)  # euclidean distance
    cv2.imshow("dist", dist.astype(np.uint8))

    markers = np.zeros(img.shape, np.int32)
    # cv2.circle(markers, (150, 150), 100, 0, -1)
    # cv2.circle(markers, (250, 250), 100, 0, -1)
    # cv2.imshow("markers", markers)
    # 扩大山脚和山顶的高差
    dist[dist > 0.0] += 1

    dist3 = np.zeros((dist.shape[0], dist.shape[1], 3), dtype=np.uint8)
    dist3[:, :, 0] = img
    dist3[:, :, 1] = img
    dist3[:, :, 2] = img

    markers[150, 150] = 1  # seed for circle one
    markers[250, 250] = 2  # seed for circle two
    markers[50, 50] = 3  # seed for background
    markers[350, 350] = 4  # seed for background

    # 执行分水岭算法
    cv2.watershed(dist3, markers)
    plt.imshow(markers)
    plt.show()
    cv2.waitKey(0)


if __name__ == "__main__":
    main()