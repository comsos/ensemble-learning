from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pickle
from flask import Flask, request
import json



path = "dataset.pkl"

# data = read_csv(path, header=0)

# with open(path, 'wb') as file:
#   pickle.dump(data, file)

with open(path, 'rb') as file:
  data = pickle.load(file)


array = data.values

X = array[:, 0:-1]
Y = array[:, -1]

seed = 10
kfold = KFold(n_splits=10, random_state=seed, shuffle=True)

num_trees = 200
max_features = None

model = RandomForestClassifier(n_estimators=num_trees,max_features=max_features)
model.fit(X,Y)

results = cross_val_score(model, X, Y, cv=kfold)
print("The Generated Classifiers:")
print(results)
print("Final Model Aggregated:")
print(results.mean())

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# val_path = "datasets/validation-cleaned.csv"
# val_data = read_csv(val_path, header=0)
# val_array = val_data.values

# X_val = val_array[:, 0:-1]
# Y_val = val_array[:, -1]

# Y_val_pred = model.predict(X_val)

# accuracy = accuracy_score(Y_val, Y_val_pred)
# precision = precision_score(Y_val, Y_val_pred, average="weighted")
# recall = recall_score(Y_val, Y_val_pred, average="weighted")
# f1 = f1_score(Y_val, Y_val_pred, average="weighted")

# print("Validation Set Metrics:")
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)

import joblib
filename = 'model.sav'
joblib.dump(model, filename)


app = Flask(__name__)

@app.route('/predict', methods = ['POST'])
def prediction_model():
  data = request.get_json()

  td = data['array']

  arr = []
  for i in td:
    arr.append(i)

  result = model.predict([arr]) 

  print(result)

  return json.dumps({"result":result[0]}) 

if __name__ == "__main__":
  app.run(port=5000, debug=True)
# print("Sample Prediction on a single test data: ")

# test1 = model.predict([[1,3,1,0,0,42,39,34]])
# print(test1)

# test1 = model.predict([[1,3,1,1,0,53,52,42]])
# print(test1)

# test1 = model.predict([[0,0,1,0,1,77,88,85]])
# print(test1)

# test1 = model.predict([[0,0,4,1,0,59,72,70]])
# print(test1)

# test1 = model.predict([[1,1,3,0,0,75,68,65]])
# print(test1)