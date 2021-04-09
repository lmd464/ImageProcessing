import cv2
import numpy as np

def my_bilinear(src, scale):
    #########################
    # TODO                  #
    # my_bilinear 완성      #
    #########################
    (h, w) = src.shape
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)

    dst = np.zeros((h_dst, w_dst))

    # bilinear interpolation 적용
    for row in range(h_dst):
        for col in range(w_dst):
            # 참고로 꼭 한줄로 구현해야 하는건 아닙니다 여러줄로 하셔도 상관없습니다.(저도 엄청길게 구현했습니다.)
            from_where_row_float = row / scale
            from_where_col_float = col / scale
            from_where_row_idx = int(from_where_row_float)
            from_where_col_idx = int(from_where_col_float)

            s = from_where_col_float - from_where_col_idx
            t = from_where_row_float - from_where_row_idx

            # 인덱스가 끝이 아닌 경우
            if from_where_row_idx < src.shape[0] - 1 and from_where_col_idx < src.shape[1] - 1:
                a = (1 - s) * (1 - t) * src[from_where_row_idx, from_where_col_idx]
                b =  s * (1 - t) * src[from_where_row_idx, from_where_col_idx + 1]
                c = (1 - s) * t * src[from_where_row_idx + 1, from_where_col_idx]
                d =  s * t * src[from_where_row_idx + 1, from_where_col_idx + 1]

            # 인덱스가 끝인 경우 : 주변값 참조에 문제가 생기므로 따로 처리
            else:
                a = (1 - s) * (1 - t) * src[from_where_row_idx, from_where_col_idx]
                if from_where_row_idx >= src.shape[0] - 1 and from_where_col_idx < src.shape[1] - 1:    # row 초과
                    b = s * (1 - t) * src[from_where_row_idx, from_where_col_idx + 1]
                    c = (1 - s) * t * src[from_where_row_idx, from_where_col_idx]
                    d = s * t * src[from_where_row_idx, from_where_col_idx + 1]
                elif from_where_row_idx < src.shape[0] - 1 and from_where_col_idx >= src.shape[1] - 1:      # col 초과
                    b = s * (1 - t) * src[from_where_row_idx, from_where_col_idx]
                    c = (1 - s) * t * src[from_where_row_idx + 1, from_where_col_idx]
                    d = s * t * src[from_where_row_idx + 1, from_where_col_idx]
                elif from_where_row_idx >= src.shape[0] - 1 and from_where_col_idx >= src.shape[1] - 1:     # row, col 초과
                    b = s * (1 - t) * src[from_where_row_idx, from_where_col_idx]
                    c = (1 - s) * t * src[from_where_row_idx, from_where_col_idx]
                    d = s * t * src[from_where_row_idx, from_where_col_idx]
            dst[row, col] = a + b + c + d


    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    scale = 1/7
    #이미지 크기 1/2배로 변경
    my_dst_mini = my_bilinear(src, scale)
    my_dst_mini = my_dst_mini.astype(np.uint8)

    #이미지 크기 2배로 변경(Lena.png 이미지의 shape는 (512, 512))
    my_dst = my_bilinear(my_dst_mini, 1/scale)
    my_dst = my_dst.astype(np.uint8)

    cv2.imshow('original', src)
    cv2.imshow('my bilinear mini', my_dst_mini)
    cv2.imshow('my bilinear', my_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


