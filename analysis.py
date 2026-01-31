from pandas import *
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

db = read_csv("train.csv") 

db = get_dummies(db,columns=["Embarked"])
db["Embarked_C"] = db["Embarked_C"].astype(int)
db["Embarked_Q"] = db["Embarked_Q"].astype(int)
db["Embarked_S"] = db["Embarked_S"].astype(int)

db = db.drop(["Ticket", "Name", "PassengerId", "Cabin"], axis = 1)

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

cat_features = ["Sex", "Alone"]

predictor = CatBoostClassifier(iterations=52, depth = 4, learning_rate=0.5, random_state=1, verbose=50, loss_function="Logloss", eval_metric="Accuracy")
predictor.fit(x_train, y_train, cat_features=cat_features, use_best_model=True,eval_set=(x_test, y_test))
y_pred = predictor.predict(x_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))

