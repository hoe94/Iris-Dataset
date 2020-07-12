import pandas as pd
#import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pickle

iris = load_iris()
dataset = pd.DataFrame(data = iris.data, columns = iris.feature_names)
dataset['target'] = pd.Series(iris.target)

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

x_train,x_test, y_train,y_test = train_test_split(x,y,test_size = 0.33, random_state = 0)

random = RandomForestClassifier(criterion= 'entropy',max_depth= 2,min_samples_leaf= 1,min_samples_split= 4,n_estimators=20)

random.fit(x_train,y_train)
random_y_pred = random.predict(x_test)

pickle.dump(random,open('C:/Users/Hoe/Desktop/Learning/Python/My Accomplishment/Iris Dataset/RandomForestClassifier.pkl','wb'))



