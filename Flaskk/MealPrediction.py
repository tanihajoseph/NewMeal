import functools

from flask import Blueprint
from flask import flash
from flask import g
from flask import redirect
from flask import render_template
from flask import request
from flask import session
from flask import url_for
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash
import pandas as pd
#from flaskr.db import get_db



import joblib as jl
from flask import Flask, request, jsonify, render_template
import numpy as np
from stldecompose.forecast_funcs import (naive,
                                         drift, 
                                         mean, 
                                         seasonal_naive)

# Seasonal Decompose
from statsmodels.tsa.seasonal import seasonal_decompose

# Holt-Winters or Triple Exponential Smoothing model
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.metrics import mean_squared_error


from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,
                                         drift, 
                                         mean, 
                                         seasonal_naive)

from datetime import datetime


bp = Blueprint("MealPrediction", __name__, url_prefix="/MealPrediction")

@bp.route("/MealPrediction", methods=("GET", "POST"))
# prediction function
def ValuePredictor(): 



    if request.method == "POST":
        Mid = request.form["Meal_ID"]
        Mid=int(Mid)
        week=request.form["Week"]
        week=int(week)


    meal_info = pd.read_csv(r'C:/Users/jtani/inventory/flask-inventory/static/meal_info.csv')
    Quantity = pd.read_csv(r'C:/Users/jtani/inventory/flask-inventory/static/QuantityRequired - Sheet1.csv')





    totalMeals=meal_info['meal_id'].unique()
    #len(totalMeals) 
    STL=[1885, 1993, 2139, 2631, 1248, 1778, 1062, 2707, 2640, 2306, 2826, 1754, 1902, 1311, 1803, 1525, 2304, 1878, 1216, 1247, 1770, 1198, 1438, 2494, 1847, 2760, 2492, 1543, 2664, 2569, 1571, 2956]
    ETS=[2539, 1207, 1230, 2322, 2290, 1727, 1109, 2126, 1971, 1558, 2581, 1962, 1445, 2444, 2867, 2704, 2577, 2490, 2104]
    Quantity=Quantity.set_index('meal_id')



    



    present=0
    Raw=[]

    try:
        for s in STL:
            if(Mid==s):
                from stldecompose import decompose, forecast
                FName="C:/Users/jtani/inventory/flask-inventory/models/STL"+str(Mid)+".xml"
                #print(FName)
                print("hi")
                model = jl.load(FName)
                print("hi")
                #print(model)
                fore=forecast(model, steps=week, fc_func=naive, seasonal=True) 
                Pred=[]
                for j in fore.values:
                    Pred.append(j[0])
                print("hi")
                RawMat=Quantity.loc[Mid]
                print("hi")
                    #print(RawMat)
                for p in range(0,len(Pred)):
                    qt='Week%s' % p
                    qt=[]
                for q in range(0,len(RawMat)):
                    rw=int(round(Pred[p]*RawMat[q]))
                    qt.append(rw)
                Raw.append(qt)
                break
        for e in ETS:
            if(Mid==e):
                FName="C:/Users/jtani/inventory/flask-inventory/models/ETS"+str(Mid)+".xml"
                model = jl.load(FName)
                Pred=[]
                Pred=model.forecast(week) 
                RawMat=Quantity.loc[Mid]
                for p in range(0,len(Pred)):
                    qt='Week%s' % p
                    qt=[]
                for q in range(0,len(RawMat)):
                    rw=int(round(Pred[p]*RawMat[q]))
                    qt.append(rw)
                Raw.append(qt)
                break
    except Exception as e:
        print("Exception",e)
        Prediction="Eneter a"
        RawMaterials="raw"
    else:



        for i in range(0,len(Pred)):
            sumi=Pred[i]+sumi
        Predicted=int(round(sumi))
        #print(Predicted)
        #print(Raw)
        Raw=np.array(Raw)

        res = np.sum(Raw, 0) 
        #print(len(Raw))
        Raw=pd.DataFrame

        #print(res)


        Prediction=Predicted
        RawMaterials=res
    

    return Prediction,RawMaterials
    return render_template("MealPrediction.html")