import cv2
import numpy as np

def dilation(B, S):
    ###############################################
    # TODO                                        #
    # dilation 함수 완성                           #
    ###############################################
    h = B.shape[0]
    w = B.shape[1]
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            if 0 <= row - 1 and row + 1 < h and 0 <= col - 1 and col + 1 < w :

                # 8 - neighborhood
                if B[row, col] == 1:
                    dst[row - 1, col - 1] = 1
                    dst[row - 1, col] = 1
                    dst[row - 1, col + 1] = 1
                    dst[row, col - 1] = 1
                    dst[row, col + 1] = 1
                    dst[row + 1, col - 1] = 1
                    dst[row + 1, col] = 1
                    dst[row + 1, col + 1] = 1

    return dst


def erosion(B, S):
    ###############################################
    # TODO                                        #
    # erosion 함수 완성                            #
    ###############################################
    h = B.shape[0]
    w = B.shape[1]
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            if 0 <= row - 1 and row + 1 < h and 0 <= col - 1 and col + 1 < w:
                if np.array_equal(B[row - 1 : row + 2, col - 1 : col + 2], S):
                    dst[row, col] = 1

    return dst


def opening(B, S):
    ###############################################
    # TODO                                        #
    # opening 함수 완성                            #
    ###############################################
    dst = dilation(erosion(B, S), S)
    return dst

def closing(B, S):
    ###############################################
    # TODO                                        #
    # closing 함수 완성                            #
    ###############################################
    dst = erosion(dilation(B, S), S)
    return dst


if __name__ == '__main__':
    B = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])

    S = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]])


    cv2.imwrite('morphology_B.png', (B*255).astype(np.uint8))

    img_dilation = dilation(B, S)
    img_dilation = (img_dilation*255).astype(np.uint8)
    print(img_dilation)
    cv2.imwrite('morphology_dilation.png', img_dilation)

    img_erosion = erosion(B, S)
    img_erosion = (img_erosion * 255).astype(np.uint8)
    print(img_erosion)
    cv2.imwrite('morphology_erosion.png', img_erosion)

    img_opening = opening(B, S)
    img_opening = (img_opening * 255).astype(np.uint8)
    print(img_opening)
    cv2.imwrite('morphology_opening.png', img_opening)

    img_closing = closing(B, S)
    img_closing = (img_closing * 255).astype(np.uint8)
    print(img_closing)
    cv2.imwrite('morphology_closing.png', img_closing)


