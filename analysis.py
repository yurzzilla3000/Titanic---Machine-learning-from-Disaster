from pandas import *

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

print(db.info())
