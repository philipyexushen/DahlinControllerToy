from common import *

@MethodInformProvider
def DahlinZeroOrder(R:np.ndarray, tPeriod:float, tObject:float, tLoop:float, Kp:float = 1.0)->(np.ndarray, np.ndarray):
    """
    大林算法->无纯迟滞系统
    :param R: 输入
    :param tPeriod: 采样周期
    :param tObject: 对象时间常数
    :param tLoop: 闭环系统时间常数
    :param Kp: 放大系数
    :return: U, Y
    """
    assert len(R.shape) == 1
    C = np.exp(-tPeriod / tLoop)
    D = np.exp(-tPeriod / tObject)
    a0 = (1 - C) / (Kp * (1 - D))
    a1 = a0 * D

    U = np.zeros(R.shape, dtype=np.float64)
    E = np.zeros(R.shape, dtype=np.float64)
    Y = np.zeros(R.shape, dtype=np.float64)
    sz = R.shape[0]

    for i in range(sz): 
        Rp1 = R[i - 1] if i - 1 >= 0 else 0  
        Ep1 = E[i - 1] if i - 1 >= 0 else 0 
        E[i] = C*Ep1 + R[i] + (-C - Kp*(1 - C))*Rp1

    for i in range(sz):
        Ep1 = E[i - 1] if i - 1 >= 0 else 0 
        Up1 = U[i - 1] if i - 1 >= 0 else 0 
        U[i] = Up1 + a0*E[i] - a1*Ep1

    for i in range(sz):
        Yp1 = Y[i - 1] if i - 1 >= 0 else 0
        Rp1 = R[i - 1] if i - 1 >= 0 else 0
        Y[i] = C*Yp1  + Kp*(1 - C)*Rp1

    return U, Y


@MethodInformProvider
def DahlinFirstOrder(R:np.ndarray, tPeriod:float, tObject:float, tLoop:float, tLag:float, K:float = 1.0)->(np.ndarray, np.ndarray):
    """
    大林算法->一阶迟滞系统
    :param R: 输入
    :param tPeriod: 采样周期
    :param tObject: 对象时间常数
    :param tLoop: 闭环系统时间常数
    :param tLag: 滞后时间
    :param K: 放大系数
    :return U, Y
    """
    C = np.exp(-tPeriod/tLoop)
    D = np.exp(-tPeriod/tObject)
    N = int(tLag / tPeriod)

    a0 = (1 - C) / (K*(1 -D))
    a1 = a0* D
    b1 = C
    b2 = 1 - C

    assert len(R.shape) == 1
    U = np.zeros(R.shape, dtype=np.float64)
    E = np.zeros(R.shape, dtype=np.float64)
    Y = np.zeros(R.shape, dtype=np.float64)
    sz = R.shape[0]

    for i in range(sz):
        Ep1 = E[i - 1] if i - 1 >= 0 else 0
        Rp1 = R[i - 1] if i - 1 >= 0 else 0 
        dataNBefore = R[i - N - 1] if i - N - 1 >= 0 else 0
        E[i] = C*Ep1 + R[i] - C*Rp1 - (1 - C)*dataNBefore

    for i in range(sz):
        Ep1 = E[i - 1] if i - 1 >= 0 else 0
        Up1 = U[i - 1] if i - 1 >= 0 else 0
        dataNBefore = U[i - N - 1] if i - N - 1 >= 0 else 0
        U[i] = b1*Up1 + b2*dataNBefore + a0*E[i] - a1*Ep1

    for i in range(sz):
        dataNBefore = R[i - N - 1] if i - N - 1 >= 0 else 0
        Yp1 = Y[i - 1] if i - 1 >= 0 else 0
        Y[i] = C*Yp1 + (1 - C)*dataNBefore

    return U, Y

@MethodInformProvider
def DahlinSecondOrder(R:np.ndarray, tPeriod:float, T1:float, T2:float, tLoop:float, tLag:float, K:float = 1.0)->(np.ndarray, np.ndarray):
    """
    大林算法->二阶迟滞系统
    :param R: 输入
    :param tPeriod: 采样周期
    :param T1: 对象时间常数1
    :param T2: 对象时间常数2
    :param tLoop: 闭环系统时间常数
    :param tLag: 滞后时间
    :param K: 放大系数
    :return U, Y:
    """
    a0 = np.exp(-tPeriod / T1)
    b0 = np.exp(-tPeriod / T2)
    c0 = np.exp(-tPeriod / tLoop)

    c1 = 1 + 1/(T2 - T1)*(T1*a0 - T2*b0)
    c2 = a0*b0 + 1/(T2 - T1)*(T1*b0 - T2*a0)
    N = int(tLag / tPeriod)
    C = np.exp(-tPeriod / tLoop)

    print(-c2/c1)

    U = np.zeros(R.shape, dtype=np.float64)
    E = np.zeros(R.shape, dtype=np.float64)
    Y = np.zeros(R.shape, dtype=np.float64)
    sz = R.shape[0]

    '''
        for i in range(sz):
        Ep1 = E[i - 1] if i - 1 >= 0 else 0
        Ep2 = E[i - 2] if i - 2 >= 0 else 0
        Rp1 = R[i - 1] if i - 1 >= 0 else 0
        Rp2 = R[i - 2] if i - 2 >= 0 else 0

        X0 = c1*R[i - N - 1] if i - N - 1 >= 0 else 0
        X1 = c2*R[i - N - 2] if i - N - 2 >= 0 else 0
        E[i] = (a0 + b0) * Ep1 - a0*b0*Ep2 + R[i] -(a0 + b0) * Rp1 + a0*b0*Rp2- K*(X0 + X1)
    '''

    for i in range(sz):
        Ep1 = E[i - 1] if i - 1 >= 0 else 0
        Rp1 = R[i - 1] if i - 1 >= 0 else 0
        dataNBefore = R[i - N - 1] if i - N - 1 >= 0 else 0
        E[i] = C * Ep1 + R[i] - C * Rp1 - (1 - C) * dataNBefore

    for i in range(sz):
        Ep1 = E[i - 1] if i - 1 >= 0 else 0
        Ep2 = E[i - 2] if i - 2 >= 0 else 0
        Up1 = U[i - 1] if i - 1 >= 0 else 0
        Up2 = U[i - 2] if i - 2 >= 0 else 0

        U0 = (1 - c0)*c1*U[i - N - 1] if i - N - 1 >=0 else 0
        U1 = (1 - c0)*c2*U[i - N - 2] if i - N - 2 >=0 else 0
        U[i] = 1/c1*((-c2 + c1*c0)*Up1 + c2*c0*Up2 + U0 + U1) \
               + K/c1*((1 - c0)*(E[i] - (a0 + b0)*Ep1 + a0*b0*Ep2))

    '''
        for i in range(sz):
        X0 = c1*R[i - N - 1] if i - N - 1 >= 0 else 0
        X1 = c2*R[i - N - 2] if i - N - 2 >= 0 else 0

        Yp1 = Y[i - 1] if i - 1 >= 0 else 0
        Yp2 = Y[i - 2] if i - 2 >= 0 else 0

        Y[i] = (a0 + b0)*Yp1 - a0*b0*Yp2 + K*(X0 + X1)
    '''
    for i in range(sz):
        dataNBefore = R[i - N - 1] if i - N - 1 >= 0 else 0
        Yp1 = Y[i - 1] if i - 1 >= 0 else 0
        Y[i] = c0*Yp1 + (1 - c0)*dataNBefore

    return U, Y


@MethodInformProvider
def DahlinSecondOrderEraseRinging(R:np.ndarray, tPeriod:float, T1:float, T2:float, tLoop:float, tLag:float, K:float = 1.0, Kp = 1.0)\
        ->(np.ndarray, np.ndarray):
    """
    大林算法->二阶迟滞系统
    :param R: 输入
    :param tPeriod: 采样周期
    :param T1: 对象时间常数1
    :param T2: 对象时间常数2
    :param tLoop: 闭环系统时间常数
    :param tLag: 滞后时间
    :param K: 放大系数
    :param Kp: 放大系数
    :return U, Y:
    """
    a0 = np.exp(-tPeriod / T1)
    b0 = np.exp(-tPeriod / T2)
    c0 = np.exp(-tPeriod / tLoop)

    c1 = 1 + 1/(T2 - T1)*(T1*a0 - T2*b0)
    c2 = a0*b0 + 1/(T2 - T1)*(T1*b0 - T2*a0)
    N = int(tLag / tPeriod)
    W = (1 - c0)/(K*(1 - a0)*(1 - b0))

    U = np.zeros(R.shape, dtype=np.float64)
    E = np.zeros(R.shape, dtype=np.float64)
    Y = np.zeros(R.shape, dtype=np.float64)
    sz = R.shape[0]

    for i in range(sz):
        Ep1 = E[i - 1] if i - 1 >= 0 else 0
        Ep2 = E[i - 2] if i - 2 >= 0 else 0
        Ep3 = E[i - 3] if i - 3 >= 0 else 0

        Rp1 = R[i - 1] if i - 1 >= 0 else 0
        Rp2 = R[i - 2] if i - 2 >= 0 else 0
        Rp3 = R[i - 3] if i - 3 >= 0 else 0

        E1 = E[i - N - 1] if i - N - 1 >= 0 else 0
        E2 = E[i - N - 2] if i - N - 2 >= 0 else 0
        E3 = E[i - N - 3] if i - N - 3 >= 0 else 0
        E4 = E[i - N - 4] if i - N - 4 >= 0 else 0

        S1 = c1 * E1
        S2 = (- c1 * (a0 + b0) + c2) * E2
        S3 = (a0 * b0 * c1 - c2 * (a0 + b0)) * E3
        S4 = a0 * b0 * c2 * E4

        R1 = -(1 - c0) * R[i - N - 1] if i - N - 1 >= 0 else 0
        R2 = (1 - c0) * (a0 + b0) * R[i - N - 2] if i - N - 2 >= 0 else 0
        R3 = -a0 * b0 * (1 - c0) * R[i - N - 3] if i - N - 3 >= 0 else 0

        E[i] = (a0 + b0 + c0) * Ep1 - (a0 * b0 + c0 * (a0 + b0)) * Ep2 + a0 * b0 * c0 * Ep3 \
               + (1 - c0) * E1 - (1 - c0) * (a0 + b0) * E2 + a0 * b0 * (1 - c0) * E3 \
               - Kp * W * (S1 + S2 + S3 + S4) \
               + R[i] - (a0 + b0 + c0) * Rp1 + (a0 * b0 + c0 * (a0 + b0)) * Rp2 - a0 * b0 * c0 * Rp3 \
               + (R1 + R2 + R3)


    for i in range(sz):
        Ep1 = E[i - 1] if i - 1 >= 0 else 0
        Ep2 = E[i - 2] if i - 2 >= 0 else 0

        Up1 = U[i - 1] if i - 1 >= 0 else 0

        U1 = U[i - N - 1] if i - N - 1 >=0 else 0

        U[i] = c0*Up1 + (1 - c0)*U1 + W*(E[i] - (a0 + b0)*Ep1 + a0*b0*Ep2)

    for i in range(sz):
        Yp1 = Y[i - 1] if i - 1 >= 0 else 0
        Yp2 = Y[i - 2] if i - 2 >= 0 else 0
        Yp3 = Y[i - 3] if i - 3 >= 0 else 0

        Y1 = Y[i - N - 1] if i - N - 1 >= 0 else 0
        Y2 = Y[i - N - 2] if i - N - 2 >= 0 else 0
        Y3 = Y[i - N - 3] if i - N - 3 >= 0 else 0
        Y4 = Y[i - N - 4] if i - N - 4 >= 0 else 0

        S1 = c1*Y1
        S2 = (- c1 * (a0 + b0) + c2) * Y2
        S3 = (a0 * b0 * c1 - c2 * (a0 + b0)) * Y3
        S4 = a0 * b0 * c2 * Y4

        X1 = c1 * R[i - N - 1] if i - N - 1 >= 0 else 0
        X2 = (- c1 * (a0 + b0) + c2) * R[i - N - 2] if i - N - 2 >= 0 else 0
        X3 = (a0 * b0 * c1 - c2 * (a0 + b0)) * R[i - N - 3] if i - N - 3 >= 0 else 0
        X4 = a0 * b0 * c2 * R[i - N - 4] if i - N - 4 >= 0 else 0

        Y[i] = (a0 + b0 + c0) * Yp1 - (a0 * b0 + c0 * (a0 + b0)) * Yp2 + a0 * b0 * c0 * Yp3 \
               + (1 - c0) * Y1 - (1 - c0) * (a0 + b0) * Y2 + a0 * b0 * (1 - c0) * Y3 \
               - Kp * W * (S1 + S2 + S3 + S4) + Kp * W * (X1 + X2 + X3 + X4)

    return U, Y


@MethodInformProvider
def PIDZeroOrder(R:np.ndarray, T1:float, Td:float, Kp:float,tPeriod:float, tObject:float, tLoop:float)->np.ndarray:
    """
    常规PID算法->无迟滞系统
    :param R:输入
    :param T1: 积分调节器参数
    :param Td:微分调节器参数
    :param Kp:比例调节器参数
    :param tPeriod: 采样周期
    :param tObject: 对象时间常数
    :param tLoop: 闭环系统时间常数
    :return:
    """
    assert len(R.shape) == 1
    C = np.exp(-tPeriod / tLoop)
    D = np.exp(-tPeriod / tObject)
    Y = np.zeros(R.shape, dtype=np.float64)
    sz = R.shape[0]

    W = Kp * (1 - C)
    K1 = tPeriod / T1
    K2 = Td / tPeriod

    for i in range(sz):
        X0 = (1 + K1 + K2) * R[i - 1] if i - 1 >= 0 else 0
        X1 = (-2*K2 - 1) * R[i -2] if i -  2 >= 0 else 0
        X2 = K2 * R[i - 3] if i - 3 >= 0 else 0

        Y0 = (1 + K1 + K2) * Y[i - 1] if i - 1 >= 0 else 0
        Y1 = (-2*K2 - 1) * Y[i - 2] if i - 2 >= 0 else 0
        Y2 = K2 * Y[i - 3] if i -  3 >= 0 else 0

        Yp1 = Y[i - 1] if i - 1 >= 0 else 0
        Yp2 = Y[i - 2] if i - 2 >= 0 else 0

        Y[i] = (D + 1)*Yp1 - D*Yp2  - W*(Y0 + Y1 + Y2) + W*(X0 + X1 + X2)

    return Y


@MethodInformProvider
def PIDFirstOrder(R:np.ndarray, T1:float, Td:float, Kp:float,tPeriod:float, tObject:float, tLoop:float, tLag:float)->np.ndarray:
    """
    常规PID算法->一阶迟滞系统
    :param R:输入
    :param T1: 积分调节器参数
    :param Td:微分调节器参数
    :param Kp:比例调节器参数
    :param tPeriod: 采样周期
    :param tObject: 对象时间常数
    :param tLoop: 闭环系统时间常数
    :param tLag:滞后时间
    :return Y:
    """
    D = np.exp(-tPeriod / tObject)
    N = int(tLag / tPeriod)

    assert len(R.shape) == 1
    Y = np.zeros(R.shape, dtype=np.float64)
    sz = R.shape[0]

    W = Kp*(1 - D)
    K1 = tPeriod / T1
    K2 = Td / tPeriod

    for i in range(sz):
        X0 = (1 + K1 + K2) * R[i - N - 1] if i - N - 1 >= 0 else 0
        X1 = (-2*K2 - 1) * R[i - N - 2] if i - N - 2 >= 0 else 0
        X2 = K2 * R[i - N - 3] if i - N - 3 >= 0 else 0

        Y0 = (1 + K1 + K2) * Y[i - N - 1] if i - N - 1 >= 0 else 0
        Y1 = (-2*K2 - 1) * Y[i - N - 2] if i - N - 2 >= 0 else 0
        Y2 = K2 * Y[i - N - 3] if i - N - 3 >= 0 else 0

        Yp1 = Y[i - 1] if i - 1 >= 0 else 0
        Yp2 = Y[i - 2] if i - 2 >= 0 else 0

        Y[i] = (D + 1)*Yp1 - D*Yp2  - W*(Y0 + Y1 + Y2) + W*(X0 + X1 + X2)

    return Y

@MethodInformProvider
def PIDSecondOrder(R:np.ndarray, T1:float, Td:float, Kp:float,tPeriod:float, tObj1:float, tObj2:float, tLoop:float, tLag:float)->np.ndarray:
    """
    常规PID算法->二阶迟滞系统
    :param R:输入
    :param T1: 积分调节器参数
    :param Td:微分调节器参数
    :param Kp:比例调节器参数
    :param tPeriod: 采样周期
    :param tObj1: 对象时间常数1
    :param tObj2: 对象时间常数2
    :param tLoop: 闭环系统时间常数
    :param tLag:滞后时间
    :return Y:
    """

    N = int(tLag / tPeriod)

    assert len(R.shape) == 1
    Y = np.zeros(R.shape, dtype=np.float64)
    sz = R.shape[0]
    K1 = tPeriod / T1
    K2 = Td / tPeriod
    a0 = np.exp(-tPeriod / tObj1)
    b0 = np.exp(-tPeriod / tObj2)

    c1 = 1 + 1/(tObj2 - tObj1)*(tObj1*a0 - tObj2*b0)
    c2 = a0*b0 + 1/(tObj2 - tObj1)*(tObj1*b0 - tObj2*a0)

    for i in range(sz):
        Y0 = c1 * (1 + K1 + K2) * Y[i - N - 1] if i - N - 1 >= 0 else 0
        Y1 = (c1 * (-2 * K2 - 1) + c2 * (1 + K1 + K2)) * Y[i - N - 2] if i - N - 2 >= 0 else 0
        Y2 = (c1 * K2 + c2 * (-2 * K2 - 1))* Y[i - N - 3] if i - N - 3 >= 0 else 0
        Y3 = K2* c2 * Y[i - N - 4] if i - N - 4 >= 0 else 0

        X0 = c1 * (1 + K1 + K2) * R[i - N - 1] if i - N - 1 >= 0 else 0
        X1 = (c1 * (-2 * K2 - 1) + c2 * (1 + K1 + K2)) * R[i - N - 2] if i - N - 2 >= 0 else 0
        X2 = (c1 * K2 + c2 * (-2 * K2 - 1)) * R[i - N - 3] if i - N - 3 >= 0 else 0
        X3 = K2 * c2 * R[i - N - 4] if i - N - 4 >= 0 else 0

        Yp1 = Y[i - 1] if i - 1 >= 0 else 0
        Yp2 = Y[i - 2] if i - 2 >= 0 else 0
        Yp3 = Y[i - 3] if i - 3 >= 0 else 0

        Y[i] = (a0 + b0 + 1)*Yp1 - (a0*b0 + a0 + b0)*Yp2 + a0*b0*Yp3 - Kp*(Y0 + Y1 + Y2 + Y3) \
                + Kp*(X0 + X1 + X2 + X3)
    return Y








