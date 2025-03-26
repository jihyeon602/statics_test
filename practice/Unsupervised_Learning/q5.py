import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

'''
-------------
t-SNE은 계산시 마다 값이 지속적으로 변경되어, 예측을 위한 학습데이터로는 사용불가
고차원 데이터 -> 2차원 축소: 시각화!를 위해 활용
-------------
'''
"""
1. 데이터를 불러오고, 
   2개의 변수만을 가질 수 있도록 고정하여 
   반환하는 함수를 구현합니다.
   
   [실습4]에서 구현한 함수를 그대로 사용할 수 있습니다. 
"""
def load_data():
    
    X, y = load_wine(return_X_y=True)
    
    column_start = 6
    X = X[:, column_start : column_start+2]
    print(X.shape)
    return X, y
    
"""
2. t-SNE를 활용하여 
   2차원 데이터를 1차원으로 축소하는 함수를 완성합니다.
"""
def tsne_data(X):
    
    tsne = TSNE(n_components=2)
    
    X_tsne = tsne.fit_transform(X)
    
    return tsne, X_tsne


def main():
    
    X, y = load_data()
    
    tsne, X_tsne = tsne_data(X)
    
    print("- original shape:   ", X.shape)
    print("- transformed shape:", X_tsne.shape)
    
    print("\n원본 데이터 X :\n", X[:5])
    print("\n차원 축소 이후 데이터 X_tsne\n",X_tsne[:5])
    
    # 시각화
    plt.figure(figsize=(8, 5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', s=50)
    plt.title("t-SNE Visualization")
    plt.colorbar()
    plt.grid(True)

    plt.savefig('q5_tsne.png')

if __name__ == '__main__':
    main()
