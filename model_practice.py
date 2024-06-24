from sklearn import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# [height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160,60, 38], [154, 54, 37], [166, 65, 40],
     [166, 655, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [171, 75, 42], [181, 75, 43]]

y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'female']

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X,y)

DTprediction = clf.predict([[190, 70, 43]])

print(DTprediction)

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.5, random_state=0)

gnb = GaussianNB()

y_pred = gnb.fit(X_train, y_train).predict(X_test)

print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))