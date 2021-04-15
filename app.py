import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import pickle
import sys
app = Flask(__name__)
import logging

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

def covid_pred(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20):
  dataset=pd.read_csv("COVID.csv")
  X=dataset.iloc[:,:-1].values
  Y = dataset.iloc[:,-1].values
            
  from sklearn.preprocessing import LabelEncoder
  encoder = LabelEncoder()
  for i in range(0,20):
    X[:,i]=encoder.fit_transform(X[:,i])
  Y = encoder.fit_transform(Y)
  X = np.asarray(X)
  Y = np.asarray(Y)
  from sklearn.linear_model import LogisticRegression
  logreg = LogisticRegression()
  logreg.fit(X, Y)
  res = logreg.predict([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20]])
  return str(res)
  
  
@app.route("/covid_predict/",methods=['POST','GET'])
def Covid_predict():
#   data=request.get_json();
#   p1 = data["p1"]
#   p2 = data["p2"]
#   p3 = data["p3"]
#   p4 = data["p4"]
#   p5 = data["p5"]
#   p6 = data["p6"]
#   p7 = data["p7"]
#   p8 = data["p8"]
#   p9 = data["p9"]
#   p10 =data["p10"]
#   p11 =data["p11"]
#   p12 =data["p12"]
#   p13 =data["p13"]
#   p14 =data["p14"]
#   p15 =data["p15"]
#   p16 =data["p16"]
#   p17 =data["p17"]
#   p18 =data["p18"]
#   p19 =data["p19"]
#   p20 =data["p20"]

  res = covid_pred(1,1,0,1,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,0)
  return str(res)
  
@app.route("/",methods=['GET'])
def default():
  return "<h1> Welcome to Cvoid predictor <h1>"

if __name__ == "__main__":
    app.run(debug = True)
