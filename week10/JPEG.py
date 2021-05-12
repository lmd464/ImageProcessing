import numpy as np
import cv2
import time

def Quantization_Luminance():
    luminance = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    return luminance

def img2block(src, n=8):
    ######################################
    # TODO                               #
    # img2block 완성                      #
    # img를 block으로 변환하기              #
    ######################################
    (h, w) = src.shape
    blocks = []
    for row in range (0, h, 8):
        for col in range(0, w, 8):
            blocks.append(src[row : row + 8, col : col + 8])

    return np.array(blocks).astype(np.float64)


def C(w, n = 8):
    if w == 0:
        return (1/n)**0.5
    else:
        return (2/n)**0.5


def DCT(block, n=8):
    ######################################
    # TODO                               #
    # DCT 완성                            #
    ######################################
    dst = np.zeros(block.shape)
    v, u = dst.shape
    y, x = np.mgrid[0:u, 0:v]

    for v_ in range(v):
        for u_ in range(u):
            tmp = block * np.cos(((2*x+1) * u_ * np.pi) / (2*n)) * np.cos(((2*y+1) * v_ * np.pi) / (2*n))
            dst[v_, u_] = C(u_, n=n) * C(v_, n=n) * np.sum(tmp)

    return np.round(dst)


def my_zigzag_scanning(block, mode='encoding', block_size=8):
    ######################################
    # TODO                               #
    # my_zigzag_scanning 완성             #
    ######################################

    # 방향 : 기본 오른쪽 위 -> (-1, +1)
    # row, col : 0 -> 초기값, 아래쪽 한칸 이동 -> (+1, 0)

    # half_reached = False -> row : 0, col : w-1 도달까지
    # row 홀수 : 안뒤집음 / 짝수 : 뒤집음
    # row 0 : 끝까지 도달, row값과 col값 바꾸고 (+1, 0)하면 다음 시작위치

    # half_reached = True -> 나머지
    # col 홀수 : 뒤집음 / 짝수 : 안뒤집음
    # col w-1 : 끝까지 도달, row값과 col값 바꾸고 (0, +1) 하면 다음 시작위치

    (row, col) = (0, 0)         # 시작 위치
    half_reached = False        # 절반을 지났는가?

    if mode == 'encoding':
        zz_scanned_block = []
        block_T = block.T
        while True:
            if len(zz_scanned_block) == block_size * block_size:
                break

            if half_reached == False:
                if (row, col) == (block_size, 0):    # 반쪽 끝에 도달 후, 뒤집힌 위치
                    half_reached = True
                    row -= 1
                    col += 1
                    continue

                if (row, col) == (0, 0):
                    zz_scanned_block.append(block[row, col])
                    row += 1

                # row 홀수 : 안뒤집음
                elif row % 2 == 1:
                    while row != 0:
                        zz_scanned_block.append(block[row, col])
                        row -= 1
                        col += 1
                    # 해당 대각선 끝까지 도달
                    zz_scanned_block.append(block[row, col])
                    row, col = col, row
                    row += 1    # 다음 시작위치까지 이동 완료

                # row 짝수 : 뒤집음
                elif row % 2 == 0:
                    while row != 0:
                        zz_scanned_block.append(block_T[row, col])
                        row -= 1
                        col += 1
                    # 해당 대각선 끝까지 도달
                    zz_scanned_block.append(block_T[row, col])
                    row, col = col, row
                    row += 1  # 다음 시작위치까지 이동 완료


            elif half_reached == True:
                if (row, col) == (block_size-1, block_size-1):
                    zz_scanned_block.append(block[row, col])

                # col 홀수 : 뒤집음
                elif col % 2 == 1:
                    while col != block_size-1:
                        zz_scanned_block.append(block_T[row, col])
                        row -= 1
                        col += 1
                    # 해당 대각선 끝까지 도달
                    zz_scanned_block.append(block_T[row, col])
                    row, col = col, row
                    col += 1    # 다음 시작위치까지 이동 완료

                # col 짝수 : 안뒤집음
                elif col % 2 == 0:
                    while col != block_size-1:
                        zz_scanned_block.append(block[row, col])
                        row -= 1
                        col += 1
                    # 해당 대각선 끝까지 도달
                    zz_scanned_block.append(block[row, col])
                    row, col = col, row
                    col += 1  # 다음 시작위치까지 이동 완료

        # 마지막부터 탐색하며, 0이 안나올때까지 지우고 EOB 남김
        prev_zz_length = len(zz_scanned_block)
        sub_idx = len(zz_scanned_block) - 1
        while sub_idx > 0 and zz_scanned_block[sub_idx] == 0:
            zz_scanned_block.pop()
            sub_idx -= 1
        if len(zz_scanned_block) < prev_zz_length:
            zz_scanned_block = zz_scanned_block + ['EOB']

        return zz_scanned_block


    elif mode == 'decoding':
        zz_restored_block = np.zeros((block_size, block_size))
        EOB_reached = False

        while True:
            if len(block) == 0:
                break

            if half_reached == False:
                if (row, col) == (block_size, 0):    # 반쪽 끝에 도달 후, 뒤집힌 위치
                    half_reached = True
                    row -= 1
                    col += 1
                    continue

                # 시작 위치
                if (row, col) == (0, 0):

                    if EOB_reached == False:
                        popped = block.pop(0)
                        if popped == 'EOB':
                            EOB_reached = True
                            zz_restored_block[row, col] = 0
                        else:
                            zz_restored_block[row, col] = popped
                    else:
                        zz_restored_block[row, col] = 0

                    row += 1

                # row 홀수 : 안뒤집음
                elif row % 2 == 1:
                    while row != 0:
                        if EOB_reached == False:
                            popped = block.pop(0)
                            if popped == 'EOB':
                                EOB_reached = True
                                zz_restored_block[row, col] = 0
                            else:
                                zz_restored_block[row, col] = popped
                        else:
                            zz_restored_block[row, col] = 0
                        row -= 1
                        col += 1

                    # 해당 대각선 끝까지 도달
                    if EOB_reached == False:
                        popped = block.pop(0)
                        if popped == 'EOB':
                            EOB_reached = True
                            zz_restored_block[row, col] = 0
                        else:
                            zz_restored_block[row, col] = popped
                    else:
                        zz_restored_block[row, col] = 0
                    row, col = col, row
                    row += 1    # 다음 시작위치까지 이동 완료

                # row 짝수 : 뒤집음
                elif row % 2 == 0:
                    while row != 0:
                        transpose_temp = zz_restored_block.T
                        if EOB_reached == False:
                            popped = block.pop(0)
                            if popped == 'EOB':
                                EOB_reached = True
                                transpose_temp[row, col] = 0
                            else:
                                transpose_temp[row, col] = popped
                        else:
                            transpose_temp[row, col] = 0
                        zz_restored_block = transpose_temp.T
                        row -= 1
                        col += 1
                    # 해당 대각선 끝까지 도달
                    transpose_temp = zz_restored_block.T
                    if EOB_reached == False:
                        popped = block.pop(0)
                        if popped == 'EOB':
                            EOB_reached = True
                            transpose_temp[row, col] = 0
                        else:
                            transpose_temp[row, col] = popped
                    else:
                        transpose_temp[row, col] = 0
                    zz_restored_block = transpose_temp.T
                    row, col = col, row
                    row += 1  # 다음 시작위치까지 이동 완료


            elif half_reached == True:

                # 끝 위치
                if (row, col) == (block_size-1, block_size-1):
                    if EOB_reached == False:
                        popped = block.pop(0)
                        if popped == 'EOB':
                            EOB_reached = True
                            zz_restored_block[row, col] = 0
                        else:
                            zz_restored_block[row, col] = popped
                    else:
                        zz_restored_block[row, col] = 0

                # col 홀수 : 뒤집음
                elif col % 2 == 1:
                    while col != block_size-1:
                        transpose_temp = zz_restored_block.T
                        if EOB_reached == False:
                            popped = block.pop(0)
                            if popped == 'EOB':
                                EOB_reached = True
                                transpose_temp[row, col] = 0
                            else:
                                transpose_temp[row, col] = popped
                        else:
                            transpose_temp[row, col] = 0
                        zz_restored_block = transpose_temp.T
                        row -= 1
                        col += 1
                    # 해당 대각선 끝까지 도달
                    transpose_temp = zz_restored_block.T
                    if EOB_reached == False:
                        popped = block.pop(0)
                        if popped == 'EOB':
                            EOB_reached = True
                            transpose_temp[row, col] = 0
                        else:
                            transpose_temp[row, col] = popped
                    else:
                        transpose_temp[row, col] = 0
                    zz_restored_block = transpose_temp.T
                    row, col = col, row
                    col += 1    # 다음 시작위치까지 이동 완료

                # col 짝수 : 안뒤집음
                elif col % 2 == 0:
                    while col != block_size-1:
                        if EOB_reached == False:
                            popped = block.pop(0)
                            if popped == 'EOB':
                                EOB_reached = True
                                zz_restored_block[row, col] = 0
                            else:
                                zz_restored_block[row, col] = popped
                        else:
                            zz_restored_block[row, col] = 0
                        row -= 1
                        col += 1
                    # 해당 대각선 끝까지 도달
                    if EOB_reached == False:
                        popped = block.pop(0)
                        if popped == 'EOB':
                            EOB_reached = True
                            zz_restored_block[row, col] = 0
                        else:
                            zz_restored_block[row, col] = popped
                    else:
                        zz_restored_block[row, col] = 0
                    row, col = col, row
                    col += 1  # 다음 시작위치까지 이동 완료

        return zz_restored_block



def DCT_inv(block, n = 8):
    ###################################################
    # TODO                                            #
    # DCT_inv 완성                                     #
    # DCT_inv 는 DCT와 다름.                            #
    ###################################################

    dst = np.zeros(block.shape)
    y, x = dst.shape
    v, u = np.mgrid[0:x, 0:y]

    C_v_u = np.zeros(block.shape)
    for v_ in range(n):
        for u_ in range(n):
            C_v_u[v_, u_] = C(v_, n=n) * C(u_, n=n)

    for y_ in range(y):
        for x_ in range(x):
            tmp = block * C_v_u * np.cos(((2 * x_ + 1) * u * np.pi) / (2 * n)) * np.cos(((2 * y_ + 1) * v * np.pi) / (2 * n))
            dst[y_, x_] = np.sum(tmp)

    return np.round(dst)


def block2img(blocks, src_shape, n = 8):
    ###################################################
    # TODO                                            #
    # block2img 완성                                   #
    # 복구한 block들을 image로 만들기                     #
    ###################################################
    (h, w) = src_shape
    # 8의 배수로 나눠떨어지지 않을 경우, padding
    if h % 8 != 0:
        h = ((h // 8) + 1) * 8
    if w % 8 != 0:
        w = ((w // 8) + 1) * 8
    src_shape = (h, w)

    integrated_blocks = np.zeros(src_shape)

    for row in range(0, h, 8):
        for col in range(0, w, 8):
            integrated_blocks[row:row+8, col:col+8] = blocks[0]
            blocks = np.delete(blocks, 0, 0)

    dst = ((integrated_blocks - np.min(integrated_blocks)) / np.max(integrated_blocks - np.min(integrated_blocks))) * 255
    dst = dst.astype(np.uint8)

    return dst


def Encoding(src, n=8):
    #################################################################################################
    # TODO                                                                                          #
    # Encoding 완성                                                                                  #
    # Encoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Encoding>')
    # img -> blocks
    blocks = img2block(src, n=n)

    #subtract 128
    blocks -= 128
    #DCT
    blocks_dct = []
    for block in blocks:
        blocks_dct.append(DCT(block, n=n))
    blocks_dct = np.array(blocks_dct)

    #Quantization + thresholding
    Q = Quantization_Luminance()
    QnT = np.round(blocks_dct / Q)

    # zigzag scanning
    zz = []
    for i in range(len(QnT)):
        zz.append(my_zigzag_scanning(QnT[i]))

    return zz, src.shape


def Decoding(zigzag, src_shape, n=8, zigzag_transpose=False):
    #################################################################################################
    # TODO                                                                                          #
    # Decoding 완성                                                                                  #
    # Decoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Decoding>')

    # zigzag scanning
    blocks = []
    for i in range(len(zigzag)):
        if zigzag_transpose == False:
            blocks.append(my_zigzag_scanning(zigzag[i], mode='decoding', block_size=n))
        else:
            blocks.append(my_zigzag_scanning(zigzag[i], mode='decoding', block_size=n).T)

    blocks = np.array(blocks)

    # Denormalizing
    Q = Quantization_Luminance()
    blocks = blocks * Q

    # inverse DCT
    blocks_idct = []
    for block in blocks:
        blocks_idct.append(DCT_inv(block, n=n))
    blocks_idct = np.array(blocks_idct)

    # add 128
    blocks_idct += 128

    # block -> img
    dst = block2img(blocks_idct, src_shape=src_shape, n=n)

    return dst



def main():
    start = time.time()
    #src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    #comp, src_shape = Encoding(src, n=8)
    #recover_img = Decoding(comp, src_shape, n=8, zigzag_transpose=False)

    # 과제의 comp.npy, src_shape.npy를 복구할 때 아래 코드 사용하기(위의 3줄은 주석처리하고, 아래 3줄은 주석 풀기)
    # zigzag transpose 옵션 : 지그재그 방향 우측부터일시 True, 하측부터일시 False
    # comp.npy : True로 해야 잘 복원됨 / Lena : False
    comp = np.load('comp.npy', allow_pickle=True)
    src_shape = np.load('src_shape.npy')
    recover_img = Decoding(comp, src_shape, n=8, zigzag_transpose=True)


    total_time = time.time() - start
    print('time : ', total_time)
    if total_time > 45:
        print('감점 예정입니다.')
    cv2.imshow('recover img', recover_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
