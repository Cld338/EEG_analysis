from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from scipy.fft import fft, ifft
import pandas as pd
import numpy as np

# =========================================================================

def bandpass_filter(data, sample_rate, cutoff_low, cutoff_high):
    fft_data = fft(data)  # 주파수 영역으로 데이터를 변환

    # 주파수 영역에서 절단 주파수(cutoff frequency)를 벗어나는 주파수를 제거
    fft_data[:int(cutoff_low * len(data) / sample_rate)] = 0
    fft_data[int(cutoff_high * len(data) / sample_rate):] = 0

    filtered_data = ifft(fft_data)  # 필터링된 데이터를 시간 영역으로 변환
    filtered_data = np.real(filtered_data)  # 실수 부분만 사용

    return filtered_data

# =========================================================================

def covariance(X):
    cov_data = X@np.transpose(X)
    covariance_matrix = cov_data/np.trace(cov_data)
    return covariance_matrix


def whitening_transform(matrix):
    # 고유값 분해를 통해 얻은 혼합 공분산 행렬에 대한 고유값 및 고유 벡터
    Lambda, U = np.linalg.eig(matrix)

    # 고유값과 고유 벡터를 통해 얻은 백색화 변환 행렬
    Q = np.diag(1/np.sqrt(Lambda))@U.T
    
    return Q


def CSP_filter(experimentNum, *classes):
    # https://blog.naver.com/PostView.nhn?isHttpsRedirect=true&blogId=timesea821&logNo=220963603607
    # 위 블로그 과정을 따라 만들었으니 참고

    classNum = len(classes)
    
    # 각 class - subject별 covariance matrix가 저장된 array
    # 나중에 subject를 experiment로 수정해야함
    # subjet 별 데이터에서 experiments를 구분해야 하는데, 언제까지가 각각의 experiments인지 모름. 찾아볼 것
    classes_covariance_matrix_per_subject = np.array([np.array([covariance(classes[i][j]) for j in range(experimentNum)]) for i in range(classNum)])

    # 각 class별 mean covariance matrix 
    classes_mean_covariance_matrix = np.zeros_like(classes_covariance_matrix_per_subject[0][0])
    for i in range(classNum):
        for j in range(experimentNum):
            classes_mean_covariance_matrix = classes_mean_covariance_matrix + classes_covariance_matrix_per_subject[i][j]
    classes_mean_covariance_matrix = classes_mean_covariance_matrix/experimentNum

    # 혼합 공분산 행렬 (전체 class의 covariance matrix의 합)
    sum_covariance_matrix = np.zeros_like(classes_mean_covariance_matrix)
    for i in range(classNum):
        sum_covariance_matrix = sum_covariance_matrix + classes_mean_covariance_matrix

    # 백색화 변환 행렬
    Q = whitening_transform(sum_covariance_matrix)
    print(Q.shape)


    # 혼합 공분산 행렬의 고유값 및 고유 벡터
    Lambda, U = np.linalg.eig(sum_covariance_matrix)


    #고유값을 기준으로 내림차순 정렬된 고유 벡터
    sorted_U = U[:, np.argsort(Lambda)[::-1]]
    
    csp_filter = np.dot(sorted_U.T, Q)
    
    return np.array(csp_filter)

# =========================================================================


class PrincipalComponentAnalysis():
    def __init__(self, n_componunts, data):
        self.analyzer = PCA(n_components=n_componunts) # 주성분을 몇개로 할지 결정
        printcipalComponents = self.analyzer.fit_transform(data)
        self.principalDf = pd.DataFrame(data=printcipalComponents, columns = [f'principal component{i+1}' for i in range(n_componunts)])

    def explained_variance_ratio_(self):
            return self.analyzer.explained_variance_ratio_



# =========================================================================

def lda(data, label, n_components, solver="svd"):
    # 2개의 클래스로 구분하기 위한 LDA 생성
    lda = LinearDiscriminantAnalysis(n_components=n_components, solver=solver)
    # fit()호출 시 target값 입력 
    lda.fit(data, label)
    transformedData = lda.transform(data)
    return transformedData