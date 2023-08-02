from dataHandler import *
from private_tool import *
from numpy import append, array


datasetA1 = MotorImageryDataset(parentDirectory(dir=currDir, separator="\\", n=1)+'/bcicompetitionIV2a/A01T.npz')
trialsByClasses = datasetA1.get_trials_by_classes([7, 9, 10])
print(trialsByClasses["left"])