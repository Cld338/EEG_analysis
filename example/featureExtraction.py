from math import log
import numpy as np
import pandas as pd

def log_variance(data, classIdx, channelNum, minTrialsNum, m=3):
    # print(minTrialsNum)

    Var = np.array([np.array([np.var(data[i][j]) for j in range(channelNum)]) for i in range(minTrialsNum)])

    VarRatio = np.array([np.array([log(Var[i][j-m]/sum(Var[i])) for j in range(m*2)]) for i in range(minTrialsNum)])

    VarRatioDF = pd.DataFrame(VarRatio)

    labelDF = pd.DataFrame([classIdx]*minTrialsNum)

    VarRatioDF = VarRatioDF.reset_index()
    VarRatioDF = pd.concat([VarRatioDF, labelDF], axis=1).iloc[:, 1:]

    VarRatioDF.columns = [f"{n}" for n in range(m*2)] + ["target"]
    return VarRatioDF