'''
Created on 2018/04/07

@author: Taichi
'''
import numpy as np

class LinearRegressGD(object):
    #初期化
    def __init__(self,eta=0.001,n_iter=20):
        self.eta=eta
        self.n_iter=n_iter
    #トレーニングを実行
    def fit(self,X,Y):
        self.w_=np.zeros(1+X.shape[1])#重みを初期化
        self.cost_=[]#コスト関数の値を初期化
        for i in range(self.n_iter):
            output=self.net_input(X)#活性化関数の出力を計算
            errors=(Y-output)#誤差を計算
            self.w_[1:] +=self.eta*X.T.dot(errors)#重みw_(1)以降を更新
            self.w_[0] +=self.eta*errors.sum()#重みw_(0)を更新
            cost=(errors**2).sum()/2.0#コスト関数を計算
            self.cost_.append(cost)#コスト計算の値を格納
        return self
    #総入力を計算
    def net_input(self,X):
        return np.dot(X,self.w_[1:])+self.w_[0]
    #予測値を計算
    def predict(self,X):
        return self.net_input(X)