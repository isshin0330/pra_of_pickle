import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df.loc[df["target"]==0, "target"] = "setosa"
df.loc[df["target"]==1, "target"] = "versicolor"
df.loc[df["target"]==2, "target"] = "verginica"

hu = "hu"

x_train,x_test,y_train,y_test = train_test_split(iris["data"],iris["target"],test_size=0.25,random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train,y_train)

with open("knn.pickle", mode="wb") as f:
    pickle.dump(knn, f)
    
hoge = "hoge"

