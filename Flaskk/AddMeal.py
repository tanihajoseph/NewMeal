import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
from flask import Blueprint
# Imports for data visualization
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import DateFormatter
from matplotlib import dates as mpld

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
# the main library has a small set of functionality
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,
                                         drift, 
                                         mean, 
                                         seasonal_naive)
import io
import joblib
from datetime import datetime


bp = Blueprint("AddMeal", __name__, url_prefix="/AddMeal")

@bp.route("/AddMeal", methods=("GET", "POST"))
def AddMeal():


    if request.method == "POST":
        Mid = request.form["Meal_ID"]
        Mid=int(Mid)
        Start= request.form["Start"]

    new_tab = pd.read_csv(path) 
    new_tab['meal_id']=Mid
    STL=[]#pass the values 
    ETS=[]#pass the vlaues

    totalMeals=new_tab['meal_id'].unique()

    totalMeals=int(totalMeals)

    new_tab=pd.DataFrame(TrainN.groupby([TrainN.meal_id,TrainN.week]).sum() )
    new_tab=new_tab.drop(['week','meal_id'], axis = 1) 


    #Generating a time series
    def timeseries_df(i,Date):
        print("Entering ts")
        ar=new_tab.loc[(i)].values
        print(len(ar))
        a=[]

        for i in range(len(ar)):
            a.append(ar[i][0])

        index = pd.date_range(start=Date, periods=len(a), freq='W-SAT')
        ts = pd.DataFrame(a, index=index, columns=['num_orders'])
        ts['num_orders']=a
        print("Leaving ts")
        return ts
    

    #Finding error for ETS Model
    def errorF(model1,model2,model3,model4,test,testlen): 
            print("Entering errorF")
            test1=test
            res1=[]
            res2=[]
            res3=[]
            res4=[]
            testlen1=testlen
            infi = float('inf')

        #Error for trend=add and seasonality=add
            res1=model1.forecast(testlen1)
            y_pred1=[]
            for i in res1:
                y_pred1.append(i)
            y_true1=[]
            for i in test1:
                y_true1.append(i[0])
            try:
                error1=mean_squared_error(y_true1, y_pred1)
            except ValueError:
                error1=infi
                pass
        #Error for trend=add and seasonality=mul
            res2=model2.forecast(testlen1)
            y_pred2=[]
            for i in res2:
                y_pred2.append(i)
            y_true2=[]
            for i in test1:
                y_true2.append(i[0])
            try:
                error2=mean_squared_error(y_true2, y_pred2)
            except ValueError:
                error2=infi
                pass
        #Error for trend=mul and seasonality=add
            res3=model3.forecast(testlen1)
            y_pred3=[]
            for i in res3:
                y_pred3.append(i)
            y_true3=[]
            for i in test1:
                y_true3.append(i[0])
            try:
                error3=mean_squared_error(y_true3, y_pred3)
            except ValueError:
                error3=infi
                pass
        #Error for trend=mul and seasonality=mul
            res4=model4.forecast(testlen1)
            y_pred4=[]
            for i in res4:
                y_pred4.append(i)
            y_true4=[]
            for i in test1:
                y_true4.append(i[0])
            try:
                error4=mean_squared_error(y_true4, y_pred4)
            except ValueError:
                error4=infi
                pass
        #Finalising Error
            emin=0
            if(error1==infi):
                emin=min(error2,error3,error4)
            elif(error2==infi):
                emin=min(error1,error3,error4)
            elif(error3==infi):
                emin=min(error1,error2,error4)
            elif(error4==infi):
                emin=min(error1,error2,error3)
            else:
                emin=min(error1,error2,error3,error4)
            if(error1==emin):
                model=1
                print("Leaving ErrorF")
                return model,error1
            elif(error2==emin):
                model=2
                print("Leaving ErrorF")
                return model,error2
            elif(error3==emin):
                model=3
                print("Leaving ErrorF")
                return model,error3
            elif(error4==emin):
                model=4
                print("Leaving ErrorF")
                return model,error4

    #building model using STL
    def stl(X,ts):
        print("Entering STL")
        from stldecompose import decompose, forecast
        train_size = int(len(X) * 0.90)
        test_size = len(X)-train_size
        train, test = ts[0:train_size], ts[train_size:len(X)]
        
        decomp = decompose(train, period=7)    
        fcast = forecast(decomp, steps=test_size, fc_func=naive, seasonal=True)

        #Error Calculation
        y_pred=[]
        for i in fcast.values:
            y_pred.append(i[0])
        y_true=[]
        for i in test.values:
            y_true.append(i[0])
        Ferror=mean_squared_error(y_true, y_pred)
        print("Leaving STL")
        return decomp,Ferror
    
    #Building Model using ETS
    def ets(X):
        print("Entering ETS")
        train_size = int(len(X) * 0.90)
        test_size = len(X)-train_size
        train, test = X[0:train_size], X[train_size:len(X)]

        #ETS_training model1
        model1 = ExponentialSmoothing(train, seasonal_periods=7, trend='add', seasonal='add',damped=True).fit(use_boxcox=True)
        #ETS_training model2
        model2 = ExponentialSmoothing(train, seasonal_periods=7, trend='add', seasonal='mul',damped=True).fit(use_boxcox=True)
        #ETS_training model3
        model3 = ExponentialSmoothing(train, seasonal_periods=7, trend='mul', seasonal='add',damped=True).fit(use_boxcox=True)
        #ETS_training model4
        model4 = ExponentialSmoothing(train, seasonal_periods=7, trend='mul', seasonal='mul',damped=True).fit(use_boxcox=True)

        #Passing to find Errors    
        model,error=errorF(model1,model2,model3,model4,test,test_size)
        print("Leaving ETS")
        return model,error
    

    #if the data doesn't have data in it   



    Start = datetime.strptime(Start, '%m-%d-%Y').date()
    cr=create_model(Start)
    cr
    def create_model(Date):
        i=totalMeals
        ts = timeseries_df(i,Date)
        X = ts.values

        #STL
        modelSTL,errorSTL=stl(X,ts)

        #ETS
        modelETS,errorETS=ets(X)

        #Comparing errors of ETS and STL
        print(errorSTL)
        print(errorETS)
        error=min(errorSTL,errorETS)
        if(error==errorSTL):
            FinalModel=modelSTL
            FModel='STL'
            print("STL")
        elif(error==errorETS):
            FinalModel=modelETS
            FModel='ETS'
            print("ETS")

        #If STL Model is appropriate
        if(FModel=='STL'):
            from stldecompose import decompose, forecast
            globals()['STL%s' % i] = FinalModel
            STL.append(i)
            FinalModel = decompose(ts, period=7)    
            joblib.dump(FinalModel, 'STL'+ str(i) +'.xml', compress=1)

        #If ETS Model is appropriate
        elif(FModel=='ETS'):
            globals()['ETS%s' % i] = FinalModel
            ETS.append(i)
            if(modelETS==1):
                FinalModel = ExponentialSmoothing(X, seasonal_periods=7, trend='add', seasonal='add',damped=True).fit(use_boxcox=True)
            if(modelETS==2):
                FinalModel = ExponentialSmoothing(X, seasonal_periods=7, trend='add', seasonal='mul',damped=True).fit(use_boxcox=True)
            if(modelETS==3):
                FinalModel = ExponentialSmoothing(X, seasonal_periods=7, trend='mul', seasonal='add',damped=True).fit(use_boxcox=True)
            if(modelETS==4):
                FinalModel = ExponentialSmoothing(X, seasonal_periods=7, trend='mul', seasonal='mul',damped=True).fit(use_boxcox=True)
            joblib.dump(FinalModel, 'ETS'+ str(i) +'.xml', compress=1)

    return render_template("AddMeal.html")