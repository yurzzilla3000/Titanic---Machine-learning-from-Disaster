from pandas import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



db = read_csv("train.csv") 


db.loc[db["Embarked"] == "C", "Embarked"] = 1
db.loc[db["Embarked"] == "Q", "Embarked"] = 2
db.loc[db["Embarked"] == "S", "Embarked"] = 3
db["Embarked"] = db["Embarked"].fillna(3)
db["Embarked"] = db["Embarked"].apply(lambda x: int(x))

db = db.drop("Ticket", axis=1)
db = db.drop("Name", axis=1)
db = db.drop("PassengerId", axis=1)
db = db.drop("Cabin", axis=1)

db["Fare"] = db["Fare"].round().astype(int)

db.loc[db["Sex"] == "female", "Sex"] = 0
db.loc[db["Sex"] == "male", "Sex"] = 1
db["Sex"] = db["Sex"].astype(int)

db["Age"].fillna(db["Age"].median(), inplace=True)
db["Age"] = db["Age"].round().astype(int)


x = db.drop("Survived", axis=1)
y = db["Survived"]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=1)

predictor = RandomForestClassifier(n_estimators=350, max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=1)

predictor.fit(x_train, y_train)

y_pred = predictor.predict(x_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))

