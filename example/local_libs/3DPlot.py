from private_tool import *
from dataAnalyzer import *
import pandas as pd

df = pd.DataFrame(np.load(currDir+"/test.npy"))
print(df)
df.columns = [f"axis{i}" for i in range(3)] + ["label"]
plotDF3D(data=df, num_of_classes=4)