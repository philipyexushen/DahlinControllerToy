# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.figure import *
from algorithm import *
from neuron import *
import numpy as np

def SetPlotDefaultProperty(yLabel:str, strTitle:str = None):
    if strTitle:
        plt.title(strTitle)
    plt.xlabel('N (t = NT)'), plt.ylabel(yLabel), plt.grid(), plt.legend()


if __name__ == "__main__":
    tPeriod = 0.5
    tObject1 = 4
    tObject2 = 6
    tLoop = 2
    tLag = 3

    # 普通阶跃信号
    R = np.ones(200, dtype=np.float64)
    for i in range(400):
        R0 = np.zeros(200, dtype=np.float64)
        R = np.append(R, R0)
        R0 = np.ones(200, dtype=np.float64)
        R = np.append(R, R0)

    dataDahlinU0, dataDahlin0 = DahlinZeroOrder(R, tPeriod, tObject1, tLoop)
    dataPID0 = PIDZeroOrder(R, 8, 0.1, 1, tPeriod, tObject1, tLoop)
    dataDahlinU1, dataDahlin1 = DahlinFirstOrder(R, tPeriod, tObject1, tLoop, tLag)
    dataPID1 = PIDFirstOrder(R, 8, 0.1, 1, tPeriod, tObject1, tLoop, tLag)
    dataDahlinU2, dataDahlin2 = DahlinSecondOrder(R, tPeriod, tObject1, tObject2, tLoop, tLag)
    dataPID2 = PIDSecondOrder(R, 8, 0.1, 1, tPeriod, tObject1,tObject2, tLoop, tLag)

    y, Y = DahlinBP(R, tPeriod, tObject1, tObject2, tLag, R.size)
    dataDahlinU2E, dataDahlin2E = DahlinSecondOrderEraseRinging(R, tPeriod, tObject1, tObject2, tLoop, tLag)

    plt.figure(1)
    plt.subplot(121)
    plt.plot(dataDahlin2E, 'r', label="DahlinErase")
    plt.plot(R, 'blue', label="H")
    plt.plot(Y[0:R.size - 1], 'orange', label="DahlinBP")
    plt.plot(dataDahlin2, 'purple', label="Dahlin")

    SetPlotDefaultProperty('Y(N)', "Dahlin Phase Step Plot")

    plt.subplot(122)
    plt.plot(y[:, 0], 'r', label="Tc")
    plt.plot(y[:, 1], 'orange', label="Kp")
    plt.plot(y[:, 2], 'blue', label="TOut")
    SetPlotDefaultProperty('Value', "Parameter")

    '''
    # 普通正弦信号
    axis_x = np.linspace(0, 20, num=1000)
    R2 = np.sin(axis_x)

    dataDahlinU0, dataDahlin0 = DahlinZeroOrder(R2, tPeriod, tObject1, tLoop)
    dataPID0 = PIDZeroOrder(R2, 8, 0.1, 1, tPeriod, tObject1, tLoop)
    dataDahlinU1, dataDahlin1 = DahlinFirstOrder(R2, tPeriod, tObject1, tLoop, tLag)
    dataPID1 = PIDFirstOrder(R2, 8, 0.1, 1, tPeriod, tObject1, tLoop, tLag)
    dataDahlinU2, dataDahlin2 = DahlinSecondOrder(R2, tPeriod, tObject1, tObject2, tLoop, tLag)
    dataPID2 = PIDSecondOrder(R2, 8, 0.1, 1, tPeriod, tObject1, tObject2, tLoop, tLag)

    y, Y = DahlinBP(R2, tPeriod, tObject1, tObject2, tLag, R2.size)

    dataDahlinU2E, dataDahlin2E = DahlinSecondOrderEraseRinging(R2, tPeriod, tObject1, tObject2, tLoop, tLag)

    plt.figure(2)
    plt.subplot(121)
    plt.plot(dataDahlin2E, 'r', label="DahlinErase")
    plt.plot(R2, 'blue', label="H")
    plt.plot(dataDahlin2, 'purple', label="Dahlin")
    plt.plot(Y[0:R2.size - 1], 'orange', label="DahlinBP")
    SetPlotDefaultProperty('Y(N)', "Dahlin Sine Plot")

    plt.subplot(122)
    plt.plot(y[:, 0], 'r', label="Tc")
    plt.plot(y[:, 1], 'orange', label="Kp")
    plt.plot(y[:, 2], 'blue', label="TOut")
    SetPlotDefaultProperty('Value', "Parameter")
    '''


    plt.show()

    '''
    plt.figure(1)

    plt.subplot(231)
    plt.plot(dataDahlin0, 'r', label="Dahlin"), plt.plot(R, 'blue', label="H"), plt.plot(dataPID0, 'orange', label="PID")
    SetPlotDefaultProperty('Y(N)', "First Order (No Delay)")

    plt.subplot(234)
    plt.plot(dataDahlinU0, 'purple', label="Dahlin Controller Output")
    SetPlotDefaultProperty('U(N)')

    plt.subplot(232)
    plt.plot(dataDahlin1, 'r', label="Dahlin"), plt.plot(R, 'blue', label="H"), plt.plot(dataPID1, 'orange', label="PID")
    SetPlotDefaultProperty('Y(N)', "First Order")

    plt.subplot(235)
    plt.plot(dataDahlinU1, 'purple', label="Dahlin Controller Output")
    SetPlotDefaultProperty('U(N)')

    plt.subplot(233)
    plt.plot(dataDahlin2, 'r', label="Dahlin"), plt.plot(R, 'blue', label="H"), plt.plot(dataPID2, 'orange', label="PID")
    SetPlotDefaultProperty('Y(N)', "Second Order")

    plt.subplot(236)
    plt.plot(dataDahlinU2, 'purple', label="Dahlin Controller Output")
    SetPlotDefaultProperty('U(N)')

    dataDahlinU2E, dataDahlin2E = DahlinSecondOrderEraseRinging(R, tPeriod, tObject1, tObject2, tLoop, tLag)
    plt.figure(2)
    plt.subplot(121)
    plt.plot(dataDahlin2E, 'r', label="DahlinErase"), plt.plot(R, 'blue', label="H"), plt.plot(dataPID2, 'orange', label="PID")
    plt.plot(dataDahlin2, 'purple', label="Dahlin")
    SetPlotDefaultProperty('Y(N)', "Hello Dahlin")

    plt.subplot(122)
    plt.plot(dataDahlinU2E, 'purple', label="Dahlin Controller Output")
    SetPlotDefaultProperty('U(N)')
    
    plt.show()
    '''





