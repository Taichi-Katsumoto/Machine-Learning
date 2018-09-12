'''
Created on 2018/09/12

@author: Taichi
'''
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(x,y,classifier,test_idx=None,resolution=0.02):
    markers=('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])

    x1_min,x1_max=x[:,0].min()-1,x[:,0].max+1
    x2_min,x2_max=x[:,1].min()-1,x[:,1].max+1

    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),
                        np.arange(x2_min,x2_max,resolution))
    Z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z=Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y==cl,0],y=x[y==cl,1],
                    alpha=0.8,c=cmap(idx),
                    marker=markers[idx],label=cl)
    if test_idx:
        x_test,y_test=x[test_idx,:],y[test_idx]
        plt.scatter(x_test[:,0],x_test[:,1],c='',alpha=1.0,linewidth=1,marker='o',s=55,lable='test_set')

link=''
link2=''
df=pd.read_csv(link,encoding='cp932')
df=pd.read_table(link,encoding='cp932')
df=pd.read_excel(link,sheetname='',encoding='cp932')
df2=pd.read_csv(link2,encoding='cp932')
df2=pd.read_table(link2,encoding='cp932')
df2=pd.read_excel(link2,sheetname='',encoding='cp932')

#二つのデータを分割したが、k分割などで分けてもok
x_train=df[''].values
y_train=df[''].values
x_test=df2[''].values
y_test=df2[''].values



tree=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)
tree.fit(x_train,y_train)

x_combined=np.vstack([x_train,x_test])
y_combined=np.hstack([y_train,y_test])
plot_decision_regions(x_combined,y_combined,classifer=tree,test_idx=range(105,150))
plt.xlabel('')
plt.ylabel('')
plt.legend(loc='upper left')
plt.show()

from sklearn.tree import export_graphviz
export_graphviz(tree,out_file='tree.dot',feature_names=["a","b"])