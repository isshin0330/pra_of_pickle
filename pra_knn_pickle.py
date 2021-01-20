import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()

import pandas as pd
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df.loc[df["target"]==0, "target"] = "setosa"
df.loc[df["target"]==1, "target"] = "versicolor"
df.loc[df["target"]==2, "target"] = "verginica"

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(iris["data"],iris["target"],test_size=0.25,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train,y_train)

import pickle
with open("knn.pickle", mode="wb") as f:
    pickle.dump(knn, f)
