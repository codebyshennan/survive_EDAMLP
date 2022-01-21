# Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pickle

def trainAndPredict(X_train, X_test, Y_train, type="all"):

  accuracy = {}
  prediction = {}

  def runSGD():
    sgd = SGDClassifier(max_iter=1000, tol=None)
    sgd.fit(X_train, Y_train)
    pickle.dump(sgd, open('./models/sgd.sav','wb'))
    prediction["Stochastic Gradient Descent"] = sgd.predict(X_test)
    accuracy["Stochastic Gradient Descent"] = round(sgd.score(X_train, Y_train) * 100, 2)

  def runRF():
    randomForest = RandomForestClassifier(n_estimators=100)
    randomForest.fit(X_train, Y_train)
    pickle.dump(randomForest, open('./models/rf.sav','wb'))
    prediction["Random Forests"] = randomForest.predict(X_test)
    accuracy["Random Forests"] = round(randomForest.score(X_train, Y_train) * 100, 2)

  def runLogReg():
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, Y_train)
    pickle.dump(logreg, open('./models/logreg.sav','wb'))
    prediction["Logistic Regression"] = logreg.predict(X_test)
    accuracy["Logistic Regression"] = round(logreg.score(X_train, Y_train) * 100, 2)

  def runKNN():
    knn = KNeighborsClassifier(n_neighbors = 3) 
    knn.fit(X_train, Y_train)
    pickle.dump(knn, open('./models/knn.sav','wb'))
    prediction["KNearestNeighbours"] = knn.predict(X_test)
    accuracy["KNearestNeighbours"] = round(knn.score(X_train, Y_train) * 100, 2)

  def runNaiveBayes():
    naiveBayes = GaussianNB() 
    naiveBayes.fit(X_train, Y_train)
    pickle.dump(naiveBayes, open('./models/naiveBayes.sav','wb'))
    prediction["Naive Bayes"] = naiveBayes.predict(X_test)  
    accuracy["Naive Bayes"] = round(naiveBayes.score(X_train, Y_train) * 100, 2)

  def runSVC():
    svc = SVC()
    svc.fit(X_train, Y_train)
    pickle.dump(svc, open('./models/svc.sav','wb'))
    prediction["Support Vector Classifier"] = svc.predict(X_test)
    accuracy["Support Vector Classifier"] = round(svc.score(X_train, Y_train) * 100, 2)
  
  def runDT():
    tree = DecisionTreeClassifier() 
    tree.fit(X_train, Y_train)
    pickle.dump(tree, open('./models/tree.sav','wb'))
    prediction["Decision Trees"] = tree.predict(X_test)  
    accuracy["Decision Trees"] = round(tree.score(X_train, Y_train) * 100, 2)

  if type == "sgd": runSGD()
  if type == "rf": runRF()
  if type == "logreg": runLogReg()
  if type == "knn": runKNN()
  if type == "naive": runNaiveBayes()
  if type == "svc": runSVC()
  if type == "dt": runDT()

  if type == "all":
    runSGD()
    runRF()
    runLogReg()
    runKNN()
    runNaiveBayes()
    runSVC()
    runDT()

  return accuracy, prediction