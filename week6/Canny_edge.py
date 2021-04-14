import cv2
import numpy as np
from my_filtering import my_filtering

# low-pass filter를 적용 후 high-pass filter적용
def apply_lowNhigh_pass_filter(src, fsize, sigma=1):
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering을 해도 되고, DoG를 이용해 한번에 해도 됨

    ###########################################
    # TODO                                    #
    # apply_lowNhigh_pass_filter 완성          #
    # Ix와 Iy 구하기                            #
    ###########################################

    # 1. Gaussian
    gaus_1D = cv2.getGaussianKernel(fsize, sigma)
    gaus_2D = np.dot(gaus_1D, gaus_1D.T)
    dst = my_filtering(src, gaus_2D)

    # 2. Sobel
    sobel_x = np.mgrid[-1:2].reshape((1, 3))  # shape : (3,) -> (1, 3)
    sobel_y = sobel_x.T

    # 3. Ix, Iy
    Ix = my_filtering(dst, sobel_x)
    Iy = my_filtering(dst, sobel_y)

    return Ix, Iy


# Ix와 Iy의 magnitude를 구함
def calcMagnitude(Ix, Iy):
    ###########################################
    # TODO                                    #
    # calcMagnitude 완성                      #
    # magnitude : ix와 iy의 magnitude         #
    ###########################################
    # Ix와 Iy의 magnitude를 계산

    magnitude = np.sqrt((Ix ** 2) + (Iy ** 2))
    return magnitude


# Ix와 Iy의 angle을 구함
def calcAngle(Ix, Iy):
    ###################################################
    # TODO                                            #
    # calcAngle 완성                                   #
    # angle     : ix와 iy의 angle                      #
    # e         : 0으로 나눠지는 경우가 있는 경우 방지용     #
    # np.arctan 사용하기(np.arctan2 사용하지 말기)        #
    ###################################################
    e = 1E-6
    # radian -> degree
    angle = np.rad2deg(np.arctan(Iy / (Ix + e)))
    return angle

# non-maximum supression 수행
def non_maximum_supression(magnitude, angle):
    ####################################################################################
    # TODO                                                                             #
    # non_maximum_supression 완성                                                       #
    # largest_magnitude     : non_maximum_supression 결과(가장 강한 edge만 남김)           #
    ####################################################################################
    (h, w) = magnitude.shape

    largest_magnitude = np.zeros((h, w))
    for row in range(1, h - 1):
        for col in range(1, w - 1):
            degree = angle[row, col]

            # angle의 범위 : -90 ~ 90
            if 0 <= degree and degree < 45:
                # 거리의 비율 : tan으로 구할 수 있음
                rate = np.tan(np.deg2rad(degree))
                left_magnitude = (rate) * magnitude[row - 1, col - 1] + (1 - rate) * magnitude[row, col - 1]
                right_magnitude = (rate) * magnitude[row + 1, col + 1] + (1 - rate) * magnitude[row, col + 1]
                # 주변값보다 크면 남기고, 아니면 지움
                if magnitude[row, col] == max(left_magnitude, magnitude[row, col], right_magnitude):
                    largest_magnitude[row, col] = magnitude[row, col]

            elif -45 > degree and degree >= -90:
                rate = np.tan(np.deg2rad(np.abs(degree)))  # 비율 : 절댓값 씌워줌
                up_magnitude = (rate) * magnitude[row - 1, col + 1] + (1 - rate) * magnitude[row - 1, col]
                down_magnitude = (rate) * magnitude[row + 1, col - 1] + (1 - rate) * magnitude[row + 1, col]
                if magnitude[row, col] == max(up_magnitude, magnitude[row, col], down_magnitude):
                    largest_magnitude[row, col] = magnitude[row, col]

            elif -45 <= degree and degree < 0:
                rate = np.tan(np.deg2rad(np.abs(degree)))  # 비율 : 절댓값 씌워줌
                left_magnitude = (rate) * magnitude[row + 1, col - 1] + (1 - rate) * magnitude[row, col - 1]
                right_magnitude = (rate) * magnitude[row - 1, col + 1] + (1 - rate) * magnitude[row, col + 1]
                if magnitude[row, col] == max(left_magnitude, magnitude[row, col], right_magnitude):
                    largest_magnitude[row, col] = magnitude[row, col]

            elif 90 >= degree and degree >= 45:
                rate = np.tan(np.deg2rad(np.abs(degree)))  # 비율 : 절댓값 씌워줌
                up_magnitude = (rate) * magnitude[row - 1, col - 1] + (1 - rate) * magnitude[row - 1, col]
                down_magnitude = (rate) * magnitude[row + 1, col + 1] + (1 - rate) * magnitude[row + 1, col]
                if magnitude[row, col] == max(up_magnitude, magnitude[row, col], down_magnitude):
                    largest_magnitude[row, col] = magnitude[row, col]

            else:
                print('error')

    largest_magnitude = (largest_magnitude / np.max(largest_magnitude) * 255).astype(np.uint8)

    return largest_magnitude



# double_thresholding 수행
def double_thresholding(src):
    dst = src.copy().astype(np.float64)
    #dst => 0 ~ 255
    dst -= dst.min()
    dst /= dst.max()
    dst *= 255
    dst = dst.astype(np.uint8)

    (h, w) = dst.shape
    high_threshold_value, _ = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)
    # high threshold value는 내장함수(otsu방식 이용)를 사용하여 구하고
    # low threshold값은 (high threshold * 0.4)로 구한다
    low_threshold_value = high_threshold_value * 0.4

    ######################################################
    # TODO                                               #
    # double_thresholding 완성                            #
    # dst     : double threshold 실행 결과 이미지           #
    ######################################################
    # 왼쪽 위부터 오른쪽 아래까지 순회 : 처음 Strong을 만나는 부분부터만 적용됨
    for row in range(h):
        for col in range(w):
            if dst[row, col] >= high_threshold_value:
                dst[row, col] = 255
            elif dst[row, col] < low_threshold_value:
                dst[row, col] = 0
            else:
                # if 문 조건 : padding 영역 고려  -> row(col) : 1 ~ width(height) - 2
                if (3 // 2) <= row and (3 // 2) <= col and row < h - (3 // 2) and col < w - (3 // 2):
                    neighbor = dst[row - 1: row + 2, col - 1: col + 2]  # 이웃 : 중앙 - 1 ~ 중앙 + 1
                    strong_found = False
                    for neighbor_row in range(3):
                        for neighbor_col in range(3):
                            # Strong edge가 neighbor에 있음
                            if neighbor[neighbor_row][neighbor_col] == 255 and strong_found == False:
                                dst[row, col] = 255
                                strong_found = True
                                break
                        if strong_found == True:
                            break

    # 오른쪽 아래부터 왼쪽 위까지 순회 : Strong으로 바뀐 부분을 기준으로 다시 끝부터 재적용
    for row in range(h - 1, -1, -1):
        for col in range(w - 1, -1, -1):
            if dst[row, col] >= high_threshold_value:
                dst[row, col] = 255
            elif dst[row, col] < low_threshold_value:
                dst[row, col] = 0
            else:
                # if 문 조건 : padding 영역 고려  -> row(col) : 1 ~ width(height) - 2
                if (3 // 2) <= row and (3 // 2) <= col and row < h - (3 // 2) and col < w - (3 // 2):
                    neighbor = dst[row - 1: row + 2, col - 1: col + 2]  # 이웃 : 중앙 - 1 ~ 중앙 + 1
                    strong_found = False
                    for neighbor_row in range(3):
                        for neighbor_col in range(3):
                            # Strong edge가 neighbor에 있음
                            if neighbor[neighbor_row][neighbor_col] == 255 and strong_found == False:
                                dst[row, col] = 255
                                strong_found = True
                                break
                        if strong_found == True:
                            break

    return dst

def my_canny_edge_detection(src, fsize=3, sigma=1):
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering
    # DoG 를 사용하여 1번 filtering
    Ix, Iy = apply_lowNhigh_pass_filter(src, fsize, sigma)

    # Ix와 Iy 시각화를 위해 임시로 Ix_t와 Iy_t 만들기
    Ix_t = np.abs(Ix)
    Iy_t = np.abs(Iy)
    Ix_t = Ix_t / Ix_t.max()
    Iy_t = Iy_t / Iy_t.max()

    cv2.imshow("Ix", Ix_t)
    cv2.imshow("Iy", Iy_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # magnitude와 angle을 구함
    magnitude = calcMagnitude(Ix, Iy)
    angle = calcAngle(Ix, Iy)

    # magnitude 시각화를 위해 임시로 magnitude_t 만들기
    magnitude_t = magnitude
    magnitude_t = magnitude_t / magnitude_t.max()
    cv2.imshow("magnitude", magnitude_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # non-maximum suppression 수행
    largest_magnitude = non_maximum_supression(magnitude, angle)

    # magnitude 시각화를 위해 임시로 magnitude_t 만들기
    largest_magnitude_t = largest_magnitude
    largest_magnitude_t = largest_magnitude_t / largest_magnitude_t.max()
    cv2.imshow("largest_magnitude", largest_magnitude_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # double thresholding 수행
    dst = double_thresholding(largest_magnitude)
    return dst

def main():
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_canny_edge_detection(src)
    cv2.imshow('original', src)
    cv2.imshow('my canny edge detection', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()