from flask import Flask,render_template,request,session,url_for,json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pygal
import random
app = Flask(__name__)
session = dict()
@app.route('/')
def call():
    return render_template("index.html")

@app.route('/upload')
def registration():
    return render_template('uploaddataset.html')

@app.route('/uploaddataset',methods=["POST","GET"])
def uploaddataset_csv_submitted():
    if request.method == "POST":
        csvfile = request.files['csvfile']
        result = csvfile.filename
        file = "E:/sales prediction with linear regression/sales/" + result
        print(file)

        session['filepath'] = file
        return render_template('uploaddataset.html',msg='sucess')
    return render_template('uploaddataset.html')
@app.route('/viewdata', methods=["POST", "GET"])
def viewdata():
    session_var_value = session.get('filepath')
    df = pd.read_csv(session_var_value)
    x = pd.DataFrame(df)
    return render_template("view.html", data=x.to_html(index=False),rows=x.shape[0])


@app.route('/preprocess', methods=["POST", "GET"])
def preprocess():
    global X,y
    session_var_value = session.get('filepath')
    df = pd.read_csv(session_var_value)
    le = LabelEncoder()
    df['casual shirt'] = le.fit_transform(df['casual shirt'])
    df['party wear blazer'] = le.fit_transform(df['party wear blazer'])
    df['casual blazer'] = le.fit_transform(df['casual blazer'])
    df['casual suit'] = le.fit_transform(df['casual suit'])
    df['party wear suit'] = le.fit_transform(df['party wear suit'])
    df['party wear jean'] = le.fit_transform(df['party wear jean'])
    df['Raymond sun glassess'] = le.fit_transform(df['Raymond sun glassess'])
    df['chudidar'] = le.fit_transform(df['chudidar'])
    df['saree'] = le.fit_transform(df['saree'])
    df['Golden Metal Bangles'] = le.fit_transform(df['Golden Metal Bangles'])
    df['DSLR camera'] = le.fit_transform(df['DSLR camera'])
    df['I Phone'] = le.fit_transform(df['I Phone'])

    file = "D:\\preprocess.csv"
    print(file)

    session['filepath'] = file
    X = df.drop(['S.no','party wear jean'],axis=1)
    y = df['party wear jean']
    x = pd.DataFrame(df)
    return render_template('preprocess.html', data=x.to_html(), msg="PREPROCESSED DATA")


@app.route('/training', methods=["POST", "GET"])
def training():
    global x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    return render_template("modelperformance.html")


@app.route('/modelperformance', methods=["POST", "GET"])
def selected_model_submitted():
    global accuracyscore,MSE
    if request.method == "POST":
        selectedalg = int(request.form['algorithm'])
        if (selectedalg == 1):
            model = LinearRegression()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            MSE=np.sqrt(metrics.mean_squared_error(y_test,y_pred))
            return render_template('modelperformance.html', msg="MSE", score=MSE,
                                   model="LinearRegression")


    return render_template('modelperformance.html')


@app.route('/prediction', methods=["POST", "GET"])
def prediction():
    global year, product
    if request.method == 'POST':
        list1 = []
        list1.extend([26, 42, 33, 45, 47, 35, 45, 25, 16, 13, 15, 2020])
        model = LinearRegression()
        model.fit(x_train, y_train)
        predi = model.predict([list1])
        product = request.form['product']
        year = int(request.form['year'])
        session['year'] = year
        session['predict'] = int(predi)
        session['product'] = product
        return render_template('prediction.html', msg='Prediction Success', predvalue=predi)
    return render_template('prediction.html')

@app.route('/graph')
def graph():
    year = session.get('year')
    p = session.get('predict')
    product = session.get('product')
    l = []
    diff = year-2016
    line_chart = pygal.Bar()
    line_chart.title = 'Sales of '+product
    line_chart.x_labels = map(str, range(2016, year+1))
    for i in range(diff):
        x = int(round(random.uniform(50.00,1000.00),2))
        l.append(x)
    else:
        l.append(p)
    line_chart.add('Results', l)
    graph_data = line_chart.render_data_uri()
    return render_template("graph.html", graph_data=graph_data)

if __name__ == ("__main__"):
    app.secret_key='..'
    app.run()