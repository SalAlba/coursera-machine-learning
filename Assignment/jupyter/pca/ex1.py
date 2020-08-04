import numpy as np

#import matplotlib as mpl
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D


from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris(as_frame=True)
X = iris.data.values
y = iris.target.values

print(iris.frame.head(2))

X_reduced_2d = PCA(n_components=2).fit_transform(X)

# import matplotlib.gridspec as gridspec
fig = plt.figure(figsize = (20,15))


# ...
ax1 = fig.add_subplot(3,3,1, projection='3d')
ax1.set_title("3D iris, first three features")
ax1.scatter(X[:, 0], X[:, 1],  X[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k', alpha=.2)


ax2 = fig.add_subplot(3,3,2, projection='3d')
ax2.set_title("3D iris, PCA 2D")
ax2.scatter(X_reduced_2d[:, 0], X_reduced_2d[:, 1], zs=0, c=y, cmap=plt.cm.Set1, edgecolor='k', alpha=.2)


ax3 = fig.add_subplot(3,3,3)
ax3.set_title("2D iris, PCA 2D")
ax3.scatter(X_reduced_2d[:, 0], X_reduced_2d[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k', alpha=.2)


plt.show()
