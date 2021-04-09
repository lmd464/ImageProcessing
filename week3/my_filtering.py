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

def my_filtering(src, ftype, fshape, pad_type='zero'):
    (h, w) = src.shape
    src_pad = my_padding(src, (fshape[0]//2, fshape[1]//2), pad_type)
    dst = np.zeros((h, w))

    if ftype == 'average':
        print('average filtering')
        ###################################################
        # TODO                                            #
        # mask 완성                                        #
        # 꼭 한줄로 완성할 필요 없음                           #
        ###################################################
        mask = np.ones(fshape) / (fshape[0] * fshape[1])

        #mask 확인
        print(mask)

    elif ftype == 'sharpening':
        print('sharpening filtering')
        ##################################################
        # TODO                                           #
        # mask 완성                                       #
        # 꼭 한줄로 완성할 필요 없음                          #
        ##################################################
        mask = np.zeros(fshape)
        mask[ fshape[0] // 2 , fshape [1] // 2 ] = 2
        mask = mask - (np.ones(fshape) / (fshape[0] * fshape[1]))

        #mask 확인
        print(mask)

    #########################################################
    # TODO                                                  #
    # dst 완성                                               #
    # dst : filtering 결과 image                             #
    # 꼭 한줄로 완성할 필요 없음                                 #
    #########################################################
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(src_pad[row:row + fshape[0], col:col + fshape[1]] * mask)

    # 오버플로우 처리
    for row in range(h):
        for col in range(w):
            if dst[row, col] > 255:
                dst[row, col] = float(255)
            elif dst[row, col] < 0:
                dst[row, col] = float(0)

    dst = (dst + 0.5).astype(np.uint8)

    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # repetition padding test
    rep_test = my_padding(src, (20,20), 'repetition')

    # 3x3 filter
    dst_average = my_filtering(src, 'average', (3,3))
    dst_sharpening = my_filtering(src, 'sharpening', (3,3))

    # 5x7 filter
    #dst_average = my_filtering(src, 'average', (5,7))
    #dst_sharpening = my_filtering(src, 'sharpening', (5,7))

    # 11x13 filter
    #dst_average = my_filtering(src, 'average', (11,13))
    #dst_sharpening = my_filtering(src, 'sharpening', (11,13))

    cv2.imshow('original', src)
    cv2.imshow('average filter', dst_average)
    cv2.imshow('sharpening filter', dst_sharpening)
    cv2.imshow('repetition padding test', rep_test.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
