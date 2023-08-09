
import numpy as np
from private_tool import *
import pandas as pd
import matplotlib.pyplot as plt
# 데이터를 DataFrame으로 생성
df = pd.DataFrame(np.load(currDir + "/dataframe.npy"))

df.columns = [f"axis{i+1}" for i in range(len(df.columns)-1)]+["label"]
print(df.columns)
# 3D scatter plot 그리기
fig = plt.figure()  
ax = fig.add_subplot(111, projection='3d')

# 클래스별로 색상을 다르게 설정
colors = ['r', 'g', 'b', 'c']
for i in range(4):  # 클래스 개수에 맞게 범위 설정
    subset = df[df['label'] == i]
    ax.scatter(subset['axis1'], subset['axis2'], subset['axis3'], c=colors[i], label=f'Class {i}', alpha=1)

ax.set_xlabel('axis 1')
ax.set_ylabel('axis 2')
ax.set_zlabel('axis 3')
ax.set_title('3D Scatter Plot of axises')
ax.legend()
plt.show()
# import numpy as np
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# clf = LinearDiscriminantAnalysis()
