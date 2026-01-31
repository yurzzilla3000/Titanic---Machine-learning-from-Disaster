from pandas import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

db = read_csv("train.csv") 

db = get_dummies(db,columns=["Embarked"])
db["Embarked_C"] = db["Embarked_C"].astype(int)
db["Embarked_Q"] = db["Embarked_Q"].astype(int)
db["Embarked_S"] = db["Embarked_S"].astype(int)

db = db.drop("Ticket", axis=1)
db = db.drop("Name", axis=1)
db = db.drop("PassengerId", axis=1)
db = db.drop("Cabin", axis=1)

db["Fare"] = (db["Fare"]*1000).astype(int)

db.loc[db["Sex"] == "female", "Sex"] = 0
db.loc[db["Sex"] == "male", "Sex"] = 1
db["Sex"] = db["Sex"].astype(int)

db["Age"] = db["Age"].fillna(db["Age"].median())
db["Age"] = db["Age"].round().astype(int)

db["FamilySize"] = db["SibSp"] + db["Parch"] + 1
db["FarePerPerson"] = db["Fare"]//db["FamilySize"]
db["Alone"] = (db["FamilySize"] == 1).astype(int)

x = db.drop("Survived", axis=1)
y = db["Survived"]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=1)

predictor = RandomForestClassifier(n_estimators=450, max_depth=11, min_samples_leaf=2, min_samples_split=4, random_state=1)
predictor.fit(x_train, y_train)
y_pred = predictor.predict(x_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))

