import cv2
import numpy as np

def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h+2*p_h, w+2*p_w))
    pad_img[p_h:p_h+h, p_w:p_w+w] = src

    return pad_img


def my_filtering(src, filter, pad_type='zero'):
    (h, w) = src.shape
    src_pad = my_padding(src, (filter.shape[0]//2, filter.shape[1]//2), pad_type)
    dst = np.zeros((h, w))
    mask = filter


    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(src_pad[row:row + filter.shape[0], col:col + filter.shape[1]] * mask)

    # 오버플로우 처리
    for row in range(h):
        for col in range(w):
            if dst[row, col] > 255:
                dst[row, col] = float(255)
            elif dst[row, col] < 0:
                dst[row, col] = float(0)

    dst = (dst + 0.5).astype(np.uint8)

    return dst
