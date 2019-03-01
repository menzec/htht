import numpy as np


def onehot(data, n):
    # buf = np.zeros(data.shape + (n, ))
    # nmsk = np.arange(data.size) * n + data.ravel()
    # buf.ravel()[nmsk] = 1
    mask0 = data + 1
    mask0[mask0 == 2] = 0
    return np.array([mask0, data])


if __name__ == "__main__":
    import cv2
    img = cv2.imread("./bag_data_msk/1.jpg", 0)
    # img = cv2.resize(img, (500, 500))
    img[img > 200] = 255
    testr = img / 255
    # testr[img > 200] = 1
    testr = testr.astype("uint8")
    testa = onehot(testr, 2)
    # testa = testa.transpose(2, 0, 1)
    cv2.imshow("img", img)
    cv2.imshow("testa0", (testa[0] * 255).astype("uint8"))
    cv2.imshow("testa1", (testa[1] * 255).astype("uint8"))
    cv2.waitKey()
