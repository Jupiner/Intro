from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

X, y = load_iris(return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
gnb = GaussianNB()
nb_y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Naive Bayes: Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != nb_y_pred).sum()))


clf = tree.DecisionTreeClassifier()

dt_y_pred = clf.fit(X,y).predict(X_test)

print("Decision Trees: Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != nb_y_pred).sum()))