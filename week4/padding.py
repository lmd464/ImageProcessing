import cv2
import numpy as np

def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h+2*p_h, w+2*p_w))
    pad_img[p_h:p_h+h, p_w:p_w+w] = src

    if pad_type == 'repetition':
        print('repetition padding')
        #########################################################
        # TODO                                                  #
        # repetition padding 완성                                #
        #########################################################

        #up
        for pad_row in range (0, p_h):
            # 위
            for pad_col in range(p_w, w + p_w):
                pad_img[pad_row, pad_col] = pad_img[p_h, pad_col]

        #down
        for pad_row in range (h + p_h, h + 2*p_h):
            # 아래
            for pad_col in range(p_w, w + p_w):
                pad_img[pad_row, pad_col] = pad_img[h + p_h - 1, pad_col]

        #left, right
        for pad_row in range (0, h + p_h*2):
            # 왼쪽
            for pad_col in range(0, p_w):
                pad_img[pad_row, pad_col] = pad_img[pad_row, p_w]

            # 오른쪽
            for pad_col in range(w + p_w, w + 2*p_w):
                pad_img[pad_row, pad_col] = pad_img[pad_row, w + p_w - 1]




    else:
        print('zero padding')

    return pad_img
