from common import *

def Sigmoid_G(x, T = 4):
    return T / (1 + np.exp(-2*x))


def SigmoidDiff_G(x, T = 4):
    return 2 * T * np.exp(-2*x) / (1 + np.exp(-2*x))**2


def Sigmoid_F(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def SigmoidDiff_F(x):
    return 4 / (np.exp(x) + np.exp(-x))**2


def _FillDahlinBPInputLayer(R:np.ndarray, Y:np.ndarray, xi:np.ndarray, a0, b0, Tc, Kp, c1, c2, tPeriod, i, N):
    """
    给xi塞初值
    """
    c0 = np.exp(-tPeriod / Tc)
    W = (1 - c0) / (Kp * (1 - a0) * (1 - b0))

    '''
    Y[i] = (a0 + b0 + c0) * Yp1 - (a0 * b0 + c0 * (a0 + b0)) * Yp2 + a0 * b0 * c0 * Yp3 \
               + (1 - c0) * Y1 - (1 - c0) * (a0 + b0) * Y2 + a0 * b0 * (1 - c0) * Y3 \
               - Kp * W * (S1 + S2 + S3 + S4) + Kp * W * (X1 + X2 + X3 + X4)
    '''
    Yp1 = Y[i - 1] if i - 1 >= 0 else 0
    Yp2 = Y[i - 2] if i - 2 >= 0 else 0
    Yp3 = Y[i - 3] if i - 3 >= 0 else 0

    Y1 = Y[i - N - 1] if i - N - 1 >= 0 else 0
    Y2 = Y[i - N - 2] if i - N - 2 >= 0 else 0
    Y3 = Y[i - N - 3] if i - N - 3 >= 0 else 0
    Y4 = Y[i - N - 4] if i - N - 4 >= 0 else 0

    S1 = c1 * Y1
    S2 = (- c1 * (a0 + b0) + c2) * Y2
    S3 = (a0 * b0 * c1 - c2 * (a0 + b0)) * Y3
    S4 = a0 * b0 * c2 * Y4

    X1 = c1 * R[i - N - 1] if i - N - 1 >= 0 else 0
    X2 = (- c1 * (a0 + b0) + c2) * R[i - N - 2] if i - N - 2 >= 0 else 0
    X3 = (a0 * b0 * c1 - c2 * (a0 + b0)) * R[i - N - 3] if i - N - 3 >= 0 else 0
    X4 = a0 * b0 * c2 * R[i - N - 4] if i - N - 4 >= 0 else 0

    xi[0] =  (a0 + b0 + c0) * Yp1
    xi[1] = - (a0 * b0 + c0 * (a0 + b0)) * Yp2
    xi[2] = + a0 * b0 * c0 * Yp3
    xi[3] = + (1 - c0) * Y1
    xi[4] = - (1 - c0) * (a0 + b0) * Y2
    xi[5] = + a0 * b0 * (1 - c0) * Y3
    xi[6] = - Kp * W * S1
    xi[7] = - Kp * W * S2
    xi[8] = - Kp * W * S3
    xi[9] = - Kp * W * S4
    xi[10] = Kp * W * X1
    xi[11] = Kp * W * X2
    xi[12] = Kp * W * X3
    xi[13] = Kp * W * X4


def _FillDahlinBPHiddenLayer(X:np.ndarray, V:np.ndarray, yThi:np.ndarray, bhi:np.ndarray, alpha_i:np.ndarray):
    alpha_i.fill(0.0)
    for k in range(bhi.size):
        for (xi, vi) in zip(X, V[k]):
            alpha_i[k] += xi*vi
        bhi[k] = Sigmoid_F(alpha_i[k] - yThi[k])


def _FillDahlinBPOutputLayer(B:np.ndarray, W:np.ndarray, tThi:np.ndarray, yi:np.ndarray, beta_i:np.ndarray):
    beta_i.fill(0.0)
    for k in range(yi.size):
        for (bi, wi) in zip(B, W[k]):
            beta_i[k] += bi*wi
        yi[k] = Sigmoid_G(beta_i[k] - tThi[k])


def Output(R:np.ndarray, Y:np.ndarray, a0, b0, Tc, Kp, c1, c2, tPeriod, i, N):
    c0 = np.exp(-tPeriod / Tc)
    W = (1 - c0) / (Kp * (1 - a0) * (1 - b0))

    Yp1 = Y[i - 1] if i - 1 >= 0 else 0
    Yp2 = Y[i - 2] if i - 2 >= 0 else 0
    Yp3 = Y[i - 3] if i - 3 >= 0 else 0

    Y1 = Y[i - N - 1] if i - N - 1 >= 0 else 0
    Y2 = Y[i - N - 2] if i - N - 2 >= 0 else 0
    Y3 = Y[i - N - 3] if i - N - 3 >= 0 else 0
    Y4 = Y[i - N - 4] if i - N - 4 >= 0 else 0

    S1 = c1 * Y1
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

    return Y[i]


@MethodInformProvider
def DahlinBP(R:np.ndarray, tPeriod:float, T1:float, T2:float, tLag:float, maximumStep):
    """
    BP网络大林算法，隐层有4个神经元
    :param R: 输入
    :param tPeriod: 采样周期
    :param T1: 对象时间常数1
    :param T2: 对象时间常数2
    :param tLag: 迟滞时间
    :param maximumStep: 训练集最大步长
    :return:
    """

    # 下面的这些暂时都是定值
    n1 = 0.01       #学习率
    n2 = 0.005       # 惯性系数
    a0 = np.exp(-tPeriod / T1)
    b0 = np.exp(-tPeriod / T2)
    c1 = 1 + 1 / (T2 - T1) * (T1 * a0 - T2 * b0)
    c2 = a0 * b0 + 1 / (T2 - T1) * (T1 * b0 - T2 * a0)
    N = int(tLag / tPeriod)

    szInput, szHidden, szOutput = 14, 4, 2
    wi = np.zeros((maximumStep, szOutput, szHidden), dtype=np.float64)
    tThi = np.zeros((maximumStep, szOutput), dtype=np.float64)
    tThi[0].fill(1)
    vi = np.zeros((maximumStep, szHidden ,szInput), dtype=np.float64)
    vi[0].fill(1)
    yThi = np.zeros((maximumStep, szInput), dtype=np.float64)
    yThi[0].fill(2)
    nnHiddenVal = np.zeros((maximumStep, szHidden), dtype=np.float64)
    nnHiddenVal.fill(0.01)
    alpha_i = np.zeros((maximumStep, szHidden), dtype=np.float64)
    xi = np.zeros(szInput, dtype=np.float64)
    yi = np.zeros((maximumStep, szOutput), dtype=np.float64)
    beta_i = np.zeros((maximumStep, szOutput), dtype=np.float64)
    sz = R.shape[0]
    assert maximumStep <= sz
    Y = np.zeros(R.shape, dtype=np.float64)
    yi[0] = (2, 1)

    for step in range(1, sz):
        Tc = yi[step - 1, 0]
        Kp = yi[step - 1, 1]

        # forward
        _FillDahlinBPInputLayer(R, Y, xi, a0, b0, Tc, Kp, c1, c2, tPeriod, step - 1, N)
        _FillDahlinBPHiddenLayer(xi, vi[step - 1], yThi[step - 1], nnHiddenVal[step], alpha_i[step])
        _FillDahlinBPOutputLayer(nnHiddenVal[step], wi[step - 1], tThi[step - 1], yi[step], beta_i[step])

        # Update Wi and tThi
        # w_hj = -n1*gj*bh + n2*pre_w_hj   thi_j = +n1*gj + n2*pre_thi_j
        sTc, sKp = yi[step]
        systemOutVal = Output(R, Y, a0, b0, sTc, sKp, c1, c2, tPeriod, step - 1, N)
        g = np.zeros(szOutput, dtype=np.float64)
        for j in range(szOutput):
            for k in range(szHidden):
                preSystemOutVal = Y[step - 2] if step - 2 >= 0 else 0
                preNeuronOut = yi[step - 1, j] if step - 1 >= 0 else 0
                preWi = wi[step - 1, j, k] if step - 1 >= 0 else 0
                preTThi = tThi[step - 1, j] if step - 1 >= 0 else 0

                g[j] = (systemOutVal - 1)*np.sign(systemOutVal - preSystemOutVal) / np.sign(preNeuronOut - yi[step, j] + 0.0000001) \
                       + SigmoidDiff_G(beta_i[step, j] - tThi[step - 1, j])

                print(f"gi {g[j]}")
                wi[step, j, k] = n2 * preWi + n1 * g[j] * nnHiddenVal[step, k]
                print(f"wi {wi[step, j, k]}")
                tThi[step, j] = n2 * preTThi - n1 * g[j]
                print(f"tThi {tThi[step, j]}")


        # Update Vi and yThi
        # y_hj = n2*pre_y_hj - n1*eh*xh   thi_j = -n2*pre_thi_j + n1*eh
        e = np.zeros(szHidden, dtype=np.float64)
        for h in range(szHidden):
            for i in range(szInput):
                preVi = vi[step - 1, h, i] if step - 1 >= 0 else 0
                preYThi = yThi[step - 1, h] if step - 1 >= 0 else 0

                s = 0
                for j in range(szOutput):
                    s += wi[step, j, h] * g[j]
                e[h] = SigmoidDiff_F(alpha_i[step, h] - yThi[step - 1, h]) * s

                vi[step, h, i] = n2 *preVi + n1*e[h] * xi[i]
                yThi[step, h] = n2 * preYThi - n1 * e[h]


    return yi[maximumStep - 1, 0], yi[maximumStep - 1, 1] , yi, Y






