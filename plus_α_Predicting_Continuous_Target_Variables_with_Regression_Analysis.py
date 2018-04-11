'''
Created on 2018/04/08

@author: Taichi
'''
#指定のフォルダを選択
#import tkinter
#from tkinter import filedialog
#root = tkinter.Tk()
#root.withdraw()
#filename=filedialog.askopenfiles(filetype= [("","*")])
filename=['https://archive.ics.uci.edu/ml/machine-learning-databases/\housing/housing.data']

#データの読み込み
#複数ファイルの場合はここでfor分を追加
import pandas as pd
df=pd.read_csv(filename[0],header=None,sep='\s+')
df.columns=["CRIM","Zn","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","RTRATIO","B","LSTAT","MEDV"]
df.head()
#データセットの可視化
import matplotlib.pyplot as plt
import seaborn as sns
#グラフのスタイル指定(whitegridは白背景グリッド、notebookはデフォルト)
sns.set(style="whitegrid",context="notebook")
cols=["LSTAT","INDUS","NOX","RM","MEDV"]
sns.pairplot(df[cols],size=2.5)
plt.show()
#sns.reset_orig()#seabornのデフォルトスタイルをリセットする。
import numpy as np
cm=np.corrcoef(df[cols].values.T)#ピアソンの積率相関係数を計算
sns.set(font_scale=1.5)#全体のスケールとは別にフォントサイズを指定
hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt=".2f",annot_kws={"size":15},yticklabels=cols,xticklabels=cols)
#第1引数の相関関係をもとにヒートマップを作成、カラーバー、データ値表示、正方形化、表示値の浮動小数点、データ値のサイズ
#行の目盛ラベル、列の目盛ラベル
plt.show()
X=df[["RM"]].values
Y=df["MEDV"].values
from LinearRegessionGD import LinearRegressGD
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()#標準化する
sc_Y=StandardScaler()
X_std=sc_X.fit_transform(X)#Xを標準化
Y_std=sc_Y.fit_transform(Y)
lr=LinearRegressGD()
lr.fit(X_std,Y_std)

#エポック数とコストの関係を表す折れ線グラフのプロット
plt.plot(range(1,lr.n_iter+1),lr.cost_)
plt.ylabel("SSE")
plt.xlabel("Epoch")
plt.show()

#ヘルパー関数の定義
def lin_regplot(X,Y,model):
    plt.scatter(X,Y,c="blue")
    plt.plot(X,model.predict(X),color="red")
    return None

lin_regplot(X_std,Y_std,lr)
plt.xlabel("Average number of rooms [RM](standardized)")
plt.xlabel("Price in $1000\'S[MEDV](standardized)")
plt.show()
#結果変数を元の尺度に戻す
num_room_std=sc_X.transform([5.0])
price_std=lr.predict(num_room_std)
print("Price in $1000's: %.3f" % sc_Y.inverse_transform(price_std))

#傾き
print("Slope: %.3f" % lr.w_[1])
#切片(標準化されたデータでは常に0)
print("Intercept: %.3f" % lr.w_[0])

#10.3.2 scikit-learnを使って回帰モデルの係数を推定する
from sklearn.linear_model import LinearRegression
slr=LinearRegression()
slr.fit(X,Y)
print("Slope: %.3f" % slr.coef_[1])
print("Intercept: %.3f" % slr.intercept_[0])
lin_regplot(X, Y, slr)
plt.xlabel("Average number of rooms [RM]")
plt.xlabel("Price in $1000\'S[MEDV]")
plt.show()
#10.4 RANSACを使ったロバスト回帰モデル
from sklearn.linear_model import RANSACRegressor
#クラスをインスタンス化
ransac=RANSACRegressor(LinearRegression(),max_trials=100,min_samples=50,residual_metric=lambda x:np.sum(np.abs(x),axis=1),
                       residual_threshold=5.0,random_state=0)
ransac.fit(X,Y)
#イテレーションの最大数を100、ランダムに選択されるサンプルの最小数を50
#redisual_metricの引数でlamba関数を指定(学習直線に対するサンプル点の縦の距離の絶対値を計算)
#residual_thresholdの引数に5.0を指定することで、学習直線に対する縦の距離が5単に距離内のサンプルだけが正常値に含むとしている。
#scikit-learnは通常はMAD推移により正常値の閾値を選択！！ただし、正常値は選択した問題に依存！RANSACの課題。


inlier_mask=ransac.inlier_mask#正常値を表す真偽値
outlier_mask=np.logical_not(inlier_mask)#外れ値を示す真偽値
line_X=np.arange(3,10,1)#3から9までの整数値を作成
line_Y_ransac=ransac.predict(line_X[:,np.newaxis])#予測値計算
plt.scatter(X[inlier_mask],Y[inlier_mask],c='blue',marker='o',label="Inliers")
plt.scatter(X[outlier_mask],Y[outlier_mask],c='lightgreen',marker='s',label="Outliers")
plt.plot(line_X,line_Y_ransac,color='red')
plt.xlabel('Average number of rooms[RM]')
plt.ylabel('Price in $1000\'s[MEDV')
plt.legend(loc='upper left')
plt.show()

print('Slope: %.3f' % ransac.estimator_.coef_[0])
print('Intercept: %.3f' % ransac.estimator_.intercept_)

#10.5線形回帰モデルの性能評価
from sklearn.cross_validation import train_test_split
X=df.iloc[:,:-1].values
Y=df["MEDV"].values

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.3, random_state=0)
slr=LinearRegression()
slr.fit(X_train,Y_train)
Y_train_pred=slr.predict(X_train)
Y_test_pred=slr.presict(X_test)
#残渣プロットはランダムに分布しているかをチェックするグラフィカルな解析に使用される。
plt.scatter(Y_train_pred,Y_train_pred-Y_train,c='blue',marker='o',label='Training data')
plt.scatter(Y_test_pred,Y_test_pred-Y_test,c='lightgreen',marker='s',label='Test data')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend(loc="upper left")
plt.hlines(y=0,xmin=-10,xmax=50,lw=2,color='red')
plt.xlim([-10,50])
plt.show()
#何らかの情報を捕捉できていない場合は残差に現れる。
#平均二乗誤差(MSE)
from sklearn.metrics import mean_squared_error
print("MSE train: %.3f, test: %.3f" % (mean_squared_error(Y_train,Y_train_pred),
                                       mean_squared_error(Y_test,Y_test_pred)))
#決定係数R^2はモデルの性能を効果的に解釈できるようにするための標準化された平均二乗誤差
#R^2=1-(SSE/SST)
#SSEは誤差平方和、SST=Σ(y^(i)-μy)^2#応答分散
#R^2=1-MSE/Var(y)
from sklearn.metrics import r2_score
print("R^2 train; %.3f , test: %.3f" % (r2_score(Y_train,Y_train_pred),
                                        r2_score(Y_test,Y_test_pred)))
#10.6 正則化手法 / リッジ回帰・LASSO・ElasticNet法
from sklearn.linear_model import Ridge
ridge=Ridge(alpha=1.0)#L2ペナルティ項の影響度合いを表す値
from sklearn.linear_model import Lasso
lasso=Lasso(alpha=1.0)
from sklearn.linear_model import ElasticNet
elasticnet=ElasticNet(alpha=1.0,l1_ratio=0.5)
#http://scikit-learn.org/stable/modules/linear_model.html



