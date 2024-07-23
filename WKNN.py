import numpy as np

def WKNN(K, rssi_fgpt, rssi_dm, rssi):
    t1, t2 = rssi_fgpt.shape
    t3 = np.tile(rssi, (t1, 1))  # 观测点的信号强度值复制，形成一个与指纹库一样行的信号强度矩阵，便于同指纹比较
    temp1 = rssi_fgpt - t3
    wknn = np.sqrt(np.sum((temp1 ** 2), axis=1))
    LMAX = np.max(wknn)
    
    wknnfmt = np.zeros((K, 3))  # 存储最邻近指纹点的坐标
    xwknn = 0  # WKNN算法坐标
    ywknn = 0
    wknnsum = 0
    
    # 获取距离最小的K个匹配网格，并把相应坐标存于wknnfmt
    for k in range(K):
        L = np.min(wknn)
        M = np.argmin(wknn)
        wknnfmt[k, 0] = rssi_dm[M, 0]
        wknnfmt[k, 1] = rssi_dm[M, 1]
        wknnfmt[k, 2] = L
        wknn[M] = LMAX
        wknnsum += 1 / L
    
    # 获取带权重的估算坐标
    for k in range(K):
        xwknn += (1 / wknnfmt[k, 2]) / wknnsum * wknnfmt[k, 0]
        ywknn += (1 / wknnfmt[k, 2]) / wknnsum * wknnfmt[k, 1]
    
    xknn = 0  # KNN算法坐标
    yknn = 0
    
    # KNN近域算法坐标
    for k in range(K):
        xknn += wknnfmt[k, 0]
        yknn += wknnfmt[k, 1]
    
    xknn /= K
    yknn /= K
    
    return xwknn, ywknn, xknn, yknn