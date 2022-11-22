import numpy as np
from flask import Flask,render_template, request
import pickle
from keras.models import load_model
app = Flask(__name__)
model = load_model('model.h5')
sc = pickle.load(open("scaler.pkl", 'rb'))  # Standard Scaler object
# model2 = pickle.load(open("model.h5", 'rb'))  # Standard Scaler object


features = ["0.799_0.201", "0.799_0.201.1",	"0.700_0.300", "0.700_0.300.1",	"0.600_0.400",
            "0.600_0.400.1", "0.501_0.499", "0.501_0.499.1", "0.400_0.600", "0.400_0.600.1"]
catagories = ["1-Octanol", "1-Propanol",
              "2-Butanol", "2-propanol", "1-isobutanol"]


@app.route('/')
def home():
    return render_template('base.html')


@app.route('/prediction')
def prediction():
    '''
    For rendering results on HTML GUI
    '''

    return render_template('index.html', features=features)


@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        results = request.form
        results = dict(results)

        inputArr = list([float(v) for v in results.values()])
        inputArr = sc.transform([inputArr])
        print(inputArr)
        index = np.argmax(model.predict(inputArr), axis=-1)[0]
        result = catagories[index]
        return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
