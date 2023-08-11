from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings

from featureExtraction import *
from dataAnalyzer import *
from private_tool import *
from dataHandler import *

warnings.filterwarnings('ignore')
    
experimentNum = 9
Experiments = [MotorImageryDataset(parentDirectory(dir=currDir, separator="\\", n=1)+'/bcicompetitionIV2a/A0'+str(i+1)+'T.npz') for i in range(experimentNum)]

channels = list(range(25))
samplingRate = 250 #Hz
trialsByClasses = np.array([i.get_trials_by_classes(channels=channels) for i in Experiments])

for i in range(len(trialsByClasses)):
    for j in trialsByClasses[i].keys():
        for k in range(len(trialsByClasses[i][j])):
            for l in range(len(trialsByClasses[i][j][k])):
                trialsByClasses[i][j][k][l] = trialsByClasses[i][j][k][l][749:1500]

channelNum = len(channels)
m = len(trialsByClasses[0]["left"])



linear_score = []
rbf_score = []
def main(experimentIdx):

    bandpassedTrialsByClasses = [{i:[[] for _ in range(channelNum)] for i in j.mi_types.values()} for j in Experiments]
    for k in range(experimentNum):
        for key in trialsByClasses[k].keys():
            for i in range(channelNum):
                bandpassedTrialsByClasses[k][key][i] = np.array([bandpass_filter(data=trialsByClasses[k][key][i][j],
                                                                    sample_rate=samplingRate,
                                                                    cutoff_low=7,
                                                                    cutoff_high=30)\
                                                    for j in range(len(trialsByClasses[k][key][i]))])

    # 현재 상태는 experiment - class - channel - trial - signal
    # class - experiment - trial - channel - signal로 수정하자
    bandpassedTrialsByClasses = {key: np.array([[bandpassedTrialsByClasses[i][key][j] for j in range(channelNum)] for i in range(experimentNum)]) for key in Experiments[0].mi_types.values()}
    # i: experiment
    # j: trial

    minTrialsNum = sorted([sorted([len(bandpassedTrialsByClasses["left"][i][0]) for i in range(experimentNum)])[0], sorted([len(bandpassedTrialsByClasses["right"][i][0]) for i in range(experimentNum)])[0], sorted([len(bandpassedTrialsByClasses["tongue"][i][0]) for i in range(experimentNum)])[0], sorted([len(bandpassedTrialsByClasses["foot"][i][0]) for i in range(experimentNum)])[0]])[0]
    bandpassedTrialsByClasses = {i:[bandpassedTrialsByClasses[i][j] for j in range(experimentNum)] for i in Experiments[0].mi_types.values()}
    minTrialsNum

    # experimentIdx = 3

    left_csp_filter = CSP_filter(np.array([i[:minTrialsNum] for i in bandpassedTrialsByClasses["left"][experimentIdx]]), np.array([j[:minTrialsNum] for i in ["right", "tongue", "foot"] for j in bandpassedTrialsByClasses[i][experimentIdx]]))
    right_csp_filter = CSP_filter(np.array([i[:minTrialsNum] for i in bandpassedTrialsByClasses["right"][experimentIdx]]), np.array([j[:minTrialsNum] for i in ["left", "right", "tongue", "foot"] for j in bandpassedTrialsByClasses[i][experimentIdx]]))
    tongue_csp_filter = CSP_filter(np.array([i[:minTrialsNum] for i in bandpassedTrialsByClasses["tongue"][experimentIdx]]), np.array([j[:minTrialsNum] for i in ["left", "tongue", "foot"] for j in bandpassedTrialsByClasses[i][experimentIdx]]))
    foot_csp_filter = CSP_filter(np.array([i[:minTrialsNum] for i in bandpassedTrialsByClasses["foot"][experimentIdx]]), np.array([j[:minTrialsNum] for i in ["left", "right", "tongue"] for j in bandpassedTrialsByClasses[i][experimentIdx]]))

    bandpassedTrials = {i:[] for i in Experiments[0].mi_types.values()}
    for i in Experiments[0].mi_types.values():
        for j in range(len(bandpassedTrialsByClasses)):
            bandpassedTrials[i]

    # CSP_passed_left_data = np.array([[bandpassedTrialsByClasses[i][j].T@left_csp_filter for j in range(minTrialsNum)] for i in range(experimentNum)])
    CSP_passed_left_data = [j[:minTrialsNum].T@left_csp_filter for j in bandpassedTrialsByClasses["left"][experimentIdx]]
    CSP_passed_right_data = [j[:minTrialsNum].T@right_csp_filter for j in bandpassedTrialsByClasses["right"][experimentIdx]]
    CSP_passed_tongue_data = [j[:minTrialsNum].T@tongue_csp_filter for j in bandpassedTrialsByClasses["tongue"][experimentIdx]]
    CSP_passed_foot_data = [j[:minTrialsNum].T@foot_csp_filter for j in bandpassedTrialsByClasses["foot"][experimentIdx]]

    len(CSP_passed_left_data)

    # bandpassedTrialsByClasses["right"][0]["right"][0]

    CSP_passed_left_data = [np.array([j.T for j in i]).T for i in CSP_passed_left_data]
    CSP_passed_right_data = [np.array([j.T for j in i]).T for i in CSP_passed_right_data]
    CSP_passed_tongue_data = [np.array([j.T for j in i]).T for i in CSP_passed_tongue_data]
    CSP_passed_foot_data = [np.array([j.T for j in i]).T for i in CSP_passed_foot_data]
    np.array(CSP_passed_left_data).shape

    CSP_passed_left_data = np.array([i.T for i in np.array(CSP_passed_left_data).T]).T
    CSP_passed_right_data = np.array([i.T for i in np.array(CSP_passed_right_data).T]).T
    CSP_passed_tongue_data = np.array([i.T for i in np.array(CSP_passed_tongue_data).T]).T
    CSP_passed_foot_data = np.array([i.T for i in np.array(CSP_passed_foot_data).T]).T
    CSP_passed_left_data.shape

    m = 12

    leftVarRatioDF = twoMDimensionalFeature(CSP_passed_left_data, channelNum, minTrialsNum, m)
    rightVarRatioDF = twoMDimensionalFeature(CSP_passed_right_data, channelNum, minTrialsNum, m)
    tongueVarRatioDF = twoMDimensionalFeature(CSP_passed_tongue_data, channelNum, minTrialsNum, m)
    footVarRatioDF = twoMDimensionalFeature(CSP_passed_foot_data, channelNum, minTrialsNum, m)

    VarRatioDF = pd.concat([leftVarRatioDF, rightVarRatioDF, tongueVarRatioDF, footVarRatioDF], axis=0)
    labelDF = pd.DataFrame([i for i in range(4) for j in range(minTrialsNum)])

    VarRatioDF = VarRatioDF.reset_index()
    VarRatioDF = pd.concat([VarRatioDF, labelDF], axis=1).iloc[:, 1:]
    VarRatioDF.columns = [f"{n}" for n in range(m*2)] + ["target"]
    VarRatioDF

    x = VarRatioDF.drop(['target'], axis=1).values
    y = VarRatioDF['target'].values # 종속변인 추출
    x = StandardScaler().fit_transform(x)
    pd.DataFrame(x)

    # plotDF3D(data=VarRatioDF, num_of_classes=4)

    # cross_validation("linear", VarRatioDF.drop(['label'], axis=1).values , labelDF)
    # cross_validation("rbf", VarRatioDF.drop(['label'], axis=1).values , labelDF)

    # n_componunts = 3
    # PCA = PrincipalComponuntAnalysis(n_componunts=n_componunts, data=x)
    # principalDF = PCA.principalDf

    # sum(PCA.explained_variance_ratio_())

    # cross_validation("linear", principalDF, labelDF)
    # cross_validation("rbf", principalDF, labelDF)

    # df = pd.concat([principalDF, labelDF], axis=1)
    # plotDF3D(data=df, num_of_classes=4)

    # np.save(currDir+"/test.npy", VarRatioDF)

    LDA_DF = pd.concat([pd.DataFrame(LDATransform(x, labelDF, 3, "eigen")), labelDF], axis=1)
    LDA_DF.columns = [f"axis{i+1}" for i in range(3)] + ["label"]

    # cross_validation("linear", LDA_DF.loc[:,:"axis3"], labelDF)
    # cross_validation("rbf", LDA_DF.loc[:,:"axis3"], labelDF)

    # plotDF3D(data=LDA_DF, num_of_classes=4)

    scaler = StandardScaler()
    scaler.fit(LDA_DF.loc[:, :"axis3"])
    LDA_DF_Scaled = pd.concat([pd.DataFrame(scaler.transform(LDA_DF.loc[:, :"axis3"])), labelDF], axis=1)
    LDA_DF_Scaled.columns = [f"axis{i+1}" for i in range(3)] + ["label"]

    linear_score.append(cross_validation("linear", LDA_DF_Scaled.loc[:,:"axis3"], labelDF))
    rbf_score.append(cross_validation("rbf", LDA_DF_Scaled.loc[:,:"axis3"], labelDF))

    # plotDF3D(data=LDA_DF, num_of_classes=4)

    # np.save(currDir+"/test.npy", LDA_DF_Scaled)
    print(experimentIdx)



for i in range(9):
    main(i)

print(linear_score)
print(sum(linear_score)/9)
print(sorted(linear_score)[4])
print(rbf_score)
print(sum(rbf_score)/9)
print(sorted(rbf_score)[4])