import numpy as np

def my_get_Gaussian2D_mask(msize, sigma=1):

    t = msize // 2
    y, x = np.mgrid[-t : t + 1, -t : t + 1]

    # 2차 gaussian mask 생성
    gaus2D = ( 1 / (2 * np.pi * (sigma ** 2)) ) * np.exp( -((x * x) + (y * y)) / (2 * (sigma ** 2) ) )

    # mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)

    return gaus2D