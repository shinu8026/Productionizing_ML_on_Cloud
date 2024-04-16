from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the model 
model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sl = request.form['sl']
    sw = request.form['sw']
    pl = request.form['pl']
    pw = request.form['pw']
    sample = np.array([int(sl), int(sw), int(pl), int(pw)]).reshape(1, -1).tolist()
    prediction = model.predict(sample)
    pred = ['setosa', 'versicolor', 'virginica'][prediction[0]]
    return render_template("index.html", value=pred)

if __name__ == '__main__':
    app.run(debug=True)
