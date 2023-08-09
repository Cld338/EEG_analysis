
import numpy as np
from private_tool import *
import pandas as pd
import matplotlib.pyplot as plt
# 데이터를 DataFrame으로 생성
df = pd.DataFrame(np.load(currDir + "/dataframe.npy"))

df.columns = [f"principal component{i+1}" for i in range(3)]+["label"]

# 3D scatter plot 그리기
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 클래스별로 색상을 다르게 설정
colors = ['r', 'g', 'b', 'c']
for i in range(4):  # 클래스 개수에 맞게 범위 설정
    subset = df[df['label'] == i]
    ax.scatter(subset['principal component1'], subset['principal component2'], subset['principal component3'], c=colors[i], label=f'Class {i}')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D Scatter Plot of Principal Components')
ax.legend()
plt.show()