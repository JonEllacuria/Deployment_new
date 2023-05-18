from flask import Flask, request, jsonify
import os
import pickle
from sklearn.model_selection import cross_val_score
import pandas as pd
import sqlite3
from sklearn.metrics import mean_absolute_error


os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a mi API del modelo advertising2"

#1. Ofrezca la predicción de ventas a partir de todos los valores de gastos en publicidad. (/v2/predict)
@app.route('/v2/predict', methods=['GET'])
def predict():
    model = pickle.load(open('data/advertising_model','rb'))

    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    if tv is None or radio is None or newspaper is None:
        return "Missing args, the input values are needed to predict"
    else:
        prediction = model.predict([[tv,radio,newspaper]])
        return "The prediction of sales investing that amount of money in TV, radio and newspaper is: " + str(round(prediction[0],2)) + 'k €'

#2. Un endpoint para almacenar nuevos registros en la base de datos que deberá estar previamente creada.(/v2/ingest_data) POST INSERT
@app.route('/v2/ingest_data_2', methods=["GET", "POST"])
def ingest_data_2():
    connection = sqlite3.connect('data/advertising.db')
    cursor = connection.cursor()
    
    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)
    sales = request.args.get('sales', None)
    
    query = f"INSERT INTO campañas (tv, radio, newspaper,sales) VALUES ({tv},{radio},{newspaper},{sales})"
    query2="SELECT * from campañas"

    if tv is None or radio is None or newspaper is None or sales is None:
        return "Missing args, the input values (tv, radio, newspaper,sales) are needed to update the data"
    else:
        cursor.execute(query).fetchall()
        result=cursor.execute(query2).fetchall()
        connection.commit()
        connection.close()
        
    
        message = "Modelo reentrenado\n"
        result_str = str(result)
        response = message + result_str
        return response
        
    

#3. Posibilidad de reentrenar de nuevo el modelo con los posibles nuevos registros que se recojan. (/v2/retrain)
@app.route('/v2/retrain', methods=["GET", "POST"])
def retrain():
    connection = sqlite3.connect('data/advertising.db')
    cursor = connection.cursor()
    
    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)
    sales = request.args.get('sales', None)
    
    query = f"INSERT INTO campañas (tv, radio, newspaper,sales) VALUES ({tv},{radio},{newspaper},{sales})"
    query2="SELECT * from campañas"
    

    if tv is None or radio is None or newspaper is None or sales is None:
        return "Missing args, the input values (tv, radio, newspaper,sales) are needed to update the data"
    else:
        cursor.execute(query).fetchall()
        result=cursor.execute(query2).fetchall()
        connection.commit()
        df = pd.read_sql_query(query2,connection)
        X=df[["TV","radio","newspaper"]]
        Y=df[["sales"]]
        connection.close()
        
        model = pickle.load(open('data/advertising_model','rb'))
        predictions = model.predict(X)
        mae = mean_absolute_error(Y, predictions)
        print(f"El MAE del modelo original es: {mae}")
        
        model2=model.fit(X,Y)
        predictions2 = model2.predict(X)
        mae2 = mean_absolute_error(Y, predictions2)
        print(f"El MAE del segundo modelo es: {mae2}")
        
        return f"El MAE del modelo original es: {mae}, y el MAE del segundo modelo es: {mae2}"


app.run()

