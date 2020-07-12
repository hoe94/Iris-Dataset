# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 12:53:34 2020

@author: Hoe
"""

from flask import Flask,render_template,request
import pickle
import sklearn

app = Flask(__name__)
model = pickle.load(open('C:/Users/Hoe/Desktop/Learning/Python/My Accomplishment/Iris Dataset/RandomForestClassifier.pkl','rb'))

@app.route('/',methods = ['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict",methods = ['POST'])
def predict():
    if request.method == "POST":
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        prediction = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
        #prediction_texts = 'The output is {}'.format(prediction[0])
        if prediction == 0:
            return render_template('index.html',prediction_texts = "This iris is belongs to Setosa (0)")
        elif prediction == 1:
            return render_template('index.html',prediction_texts = "This iris is belongs to Versicolour (1)")
        else:
            return render_template('index.html',prediction_texts = "This iris is belongs to Virginica (2)")
    else:
        return render_template('index.html')
        #print('Error Msg')

if __name__ == "__main__":
    app.run(debug=True)