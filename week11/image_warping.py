import numpy as np
import cv2

def forward(src, M, fit=False):
    #####################################################
    # TODO                                              #
    # forward 완성                                      #
    #####################################################
    print('< forward >')
    print('M')
    print(M)
    h, w = src.shape

    if fit:
        # *** DST 크기 조절 ***
        # 꼭짓점 4개의 나중위치만 구하면 M 적용 후의 전체적인 크기 정보 알 수 있을 것
        # M 적용할 때엔 col, row , 1 로 순서바꾸기 필요

        # 꼭짓점 4개 위치
        left_top = np.array([[0], [0], [1]])
        right_top = np.array([[w-1], [0], [1]])
        left_bot = np.array([[0], [h-1], [1]])
        right_bot = np.array([[w-1], [h-1], [1]])

        # 꼭짓점 4개의 M 적용 후 위치
        left_top_after_M = np.dot(M, left_top)
        right_top_after_M = np.dot(M, right_top)
        left_bot_after_M = np.dot(M, left_bot)
        right_bot_after_M = np.dot(M, right_bot)

        # M 적용 후의 최대 / 최소 인덱스들
        height_max = np.ceil(max(left_top_after_M[1], right_top_after_M[1], left_bot_after_M[1], right_bot_after_M[1])[0]).astype(int)
        height_min = np.floor(min(left_top_after_M[1], right_top_after_M[1], left_bot_after_M[1], right_bot_after_M[1])[0]).astype(int)
        width_max = np.ceil(max(left_top_after_M[0], right_top_after_M[0], left_bot_after_M[0], right_bot_after_M[0])[0]).astype(int)
        width_min = np.floor(min(left_top_after_M[0], right_top_after_M[0], left_bot_after_M[0], right_bot_after_M[0])[0]).astype(int)

        # 최대 / 최소 인덱스들을 이용하여 DST의 크기 구함
        h_after_M = height_max - height_min
        w_after_M = width_max - width_min
        dst = np.zeros((h_after_M + 1, w_after_M + 1))

        # FIT을 위한 평행이동량
        trans_row = height_min
        trans_col = width_min

        # 중첩 횟수
        N = np.zeros(dst.shape)


        # 원본에 대하여 탐색
        for row in range(h):
            for col in range(w):

                # 적용할 대상 좌표
                P = np.array([
                    [col],
                    [row],
                    [1]
                ])

                # 목적지 좌표 연산
                P_dst = np.dot(M, P)
                dst_col = P_dst[0][0]
                dst_row = P_dst[1][0]

                # 목적지 주변 4개의 좌표
                dst_col_right = int(np.ceil(dst_col)) - trans_col
                dst_col_left = int(dst_col) - trans_col
                dst_row_bottom = int(np.ceil(dst_row)) - trans_row
                dst_row_top = int(dst_row) - trans_row

                # TOP LEFT
                dst[dst_row_top, dst_col_left] += src[row, col]
                N[dst_row_top, dst_col_left] += 1

                # TOP RIGHT
                if dst_col_right != dst_col_left:
                    dst[dst_row_top, dst_col_right] += src[row, col]
                    N[dst_row_top, dst_col_right] += 1

                # BOTTOM LEFT
                if dst_row_bottom != dst_row_top:
                    dst[dst_row_bottom, dst_col_left] += src[row, col]
                    N[dst_row_bottom, dst_col_left] += 1

                # BOTTOM RIGHT
                if dst_col_right != dst_col_left and dst_row_bottom != dst_row_top:
                    dst[dst_row_bottom, dst_col_right] += src[row, col]
                    N[dst_row_bottom, dst_col_right] += 1
    else:
        dst = np.zeros((h, w))
        N = np.zeros(dst.shape)

        # 원본에 대하여 탐색
        for row in range(h):
            for col in range(w):

                # 적용할 대상 좌표
                P = np.array([
                    [col],
                    [row],
                    [1]
                ])

                # 목적지 좌표 연산
                P_dst = np.dot(M, P)
                dst_col = P_dst[0][0]
                dst_row = P_dst[1][0]

                # 목적지 주변 4개의 좌표
                dst_col_right = int(np.ceil(dst_col))
                dst_col_left = int(dst_col)
                dst_row_bottom = int(np.ceil(dst_row))
                dst_row_top = int(dst_row)

                # 범위 벗어나는거 버림
                if 0 <= dst_row_top <= h - 1 and 0 <= dst_row_bottom <= h - 1 and \
                    0 <= dst_col_left <= h - 1 and 0 <= dst_col_right <= h - 1:

                    # TOP LEFT
                    dst[dst_row_top, dst_col_left] += src[row, col]
                    N[dst_row_top, dst_col_left] += 1

                    # TOP RIGHT
                    if dst_col_right != dst_col_left:
                        dst[dst_row_top, dst_col_right] += src[row, col]
                        N[dst_row_top, dst_col_right] += 1

                    # BOTTOM LEFT
                    if dst_row_bottom != dst_row_top:
                        dst[dst_row_bottom, dst_col_left] += src[row, col]
                        N[dst_row_bottom, dst_col_left] += 1

                    # BOTTOM RIGHT
                    if dst_col_right != dst_col_left and dst_row_bottom != dst_row_top:
                        dst[dst_row_bottom, dst_col_right] += src[row, col]
                        N[dst_row_bottom, dst_col_right] += 1

    dst = np.round(dst / (N + 1E-6))
    dst = dst.astype(np.uint8)
    return dst




def backward(src, M, fit=False):
    #####################################################
    # TODO                                              #
    # backward 완성                                      #
    #####################################################
    print('< backward >')
    print('M')
    print(M)
    h, w = src.shape

    # Backward : M의 역행렬로 계산
    M_inv = np.linalg.inv(M)
    print('M inv')
    print(M_inv)

    # *** DST 크기 조절 ***
    # 꼭짓점 4개의 나중위치만 구하면 M 적용 후의 전체적인 크기 정보 알 수 있을 것
    # M 적용할 때엔 col, row , 1 로 순서바꾸기 필요

    if fit:
        # 꼭짓점 4개 위치
        left_top = np.array([[0], [0], [1]])
        right_top = np.array([[w - 1], [0], [1]])
        left_bot = np.array([[0], [h - 1], [1]])
        right_bot = np.array([[w - 1], [h - 1], [1]])

        # 꼭짓점 4개의 M 적용 후 위치
        left_top_after_M = np.dot(M, left_top)
        right_top_after_M = np.dot(M, right_top)
        left_bot_after_M = np.dot(M, left_bot)
        right_bot_after_M = np.dot(M, right_bot)

        # M 적용 후의 최대 / 최소 인덱스들
        height_max = np.ceil(max(left_top_after_M[1], right_top_after_M[1], left_bot_after_M[1], right_bot_after_M[1])[0]).astype(int)
        height_min = np.floor(min(left_top_after_M[1], right_top_after_M[1], left_bot_after_M[1], right_bot_after_M[1])[0]).astype(int)
        width_max = np.ceil(max(left_top_after_M[0], right_top_after_M[0], left_bot_after_M[0], right_bot_after_M[0])[0]).astype(int)
        width_min = np.floor(min(left_top_after_M[0], right_top_after_M[0], left_bot_after_M[0], right_bot_after_M[0])[0]).astype(int)

        # 최대 / 최소 인덱스들을 이용하여 DST의 크기 구함
        h_after_M = height_max - height_min
        w_after_M = width_max - width_min
        dst = np.zeros((h_after_M + 1, w_after_M + 1))

        # FIT을 위한 평행이동량
        trans_row = height_min
        trans_col = width_min

    else:
        trans_row = 0
        trans_col = 0
        dst = np.zeros((h, w))

    h_dst, w_dst = dst.shape
    for row in range(trans_row, trans_row + h_dst):
        for col in range(trans_col, trans_col + w_dst):

            P_dst = np.array([
                [col],
                [row],
                [1]
            ])

            P = np.dot(M_inv, P_dst)
            src_col = P[0, 0]
            src_row = P[1, 0]

            src_col_right = int(np.ceil(src_col))
            src_col_left = int(src_col)
            src_row_bottom = int(np.ceil(src_row))
            src_row_top = int(src_row)


            # src의 범위 벗어난 곳에서 가져오는 경우 제외
            if not (0 <= src_row_top < h and 0 <= src_row_bottom < h and
                    0 <= src_col_left < w and 0 <= src_col_right < w):
                continue

            s = src_col - src_col_left
            t = src_row - src_row_top

            intensity = (1-s) * (1-t) * src[src_row_top, src_col_left] \
                        + s * (1-t) * src[src_row_top, src_col_right] \
                        + (1-s) * t * src[src_row_bottom, src_col_left] \
                        + s * t * src[src_row_bottom, src_col_right]

            dst[row - trans_row, col - trans_col] = intensity

    dst = dst.astype(np.uint8)
    return dst


def main():
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    #####################################################
    # TODO                                              #
    # M 완성                                             #
    # M_tr, M_sc ... 등등 모든 행렬 M 완성하기              #
    #####################################################
    # translation
    M_tr = np.array([
        [1, 0, -30],
        [0, 1, 50],
        [0, 0, 1]
    ])

    # scaling
    M_sc = np.array([
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 1]
    ])

    # rotation
    degree = -20
    M_ro = np.array([
        [np.cos(np.deg2rad(degree)), -np.sin(np.deg2rad(degree)), 0],
        [np.sin(np.deg2rad(degree)), np.cos(np.deg2rad(degree)), 0],
        [0, 0, 1]
    ])

    # shearing
    M_sh = np.array([
        [1, 0.2, 0],
        [0.2, 1, 0],
        [0, 0, 1]
    ])

    # rotation -> translation -> Scale -> Shear
    M = np.dot(M_sh, np.dot(M_sc, np.dot(M_tr, M_ro)))


    # fit이 True인 경우와 False인 경우 다 해야 함.
    fit = True
    # forward
    dst_for = forward(src, M, fit=fit)
    dst_for2 = forward(dst_for, np.linalg.inv(M), fit=fit)

    # backward
    dst_back = backward(src, M, fit=fit)
    dst_back2 = backward(dst_back, np.linalg.inv(M), fit=fit)

    cv2.imshow('original', src)
    #cv2.imshow('forward', dst_for)     # 추가 : 중간과정
    cv2.imshow('forward2', dst_for2)
    #cv2.imshow('backward', dst_back)    # 추가 : 중간과정
    cv2.imshow('backward2', dst_back2)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ =='__main__':
    main()