from math import log
import numpy as np
import pandas as pd

def twoTimesMDimensionalFeature(data, classIdx :int, channelNum :int, minTrialsNum :int, m :int=3) -> np.ndarray:
    # doi.org/10.5626/JOK.2017.44.6.587
    """EEG 신호에 대해 2m 차원 특징 벡터 반환
    
    Args:
        data (_type_): 각각의 Trial에 대한 채널들의 EEG 데이터
        classIdx (int): 해당 클래스의 인덱스
        channelNum (int): 채널 갯수
        minTrialsNum (int): 클래스 내에서 가장 적은 Trials의 수
        m (int, optional): 차원 파라미터. Defaults to 3.

    Returns:
        np.ndarray: 특징 벡터
    """
    Var = np.array([np.array([np.var(data[i][j]) for j in range(channelNum)]) for i in range(minTrialsNum)])
    VarRatio = np.array([np.array([log(Var[i][j-m]/sum(Var[i])) for j in range(m*2)]) for i in range(minTrialsNum)])
    VarRatioDF = pd.DataFrame(VarRatio)

    return VarRatioDF


# def RMS(data):
