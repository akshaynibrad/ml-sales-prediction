import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

FORM_HTML = """
<!DOCTYPE html>
<html>
<head><title>Sales Prediction</title></head>
<body>
  <h2>Enter Features for Sales Prediction</h2>
  <form action="/predict" method="post">
    <label for="features">Features (comma separated):</label><br/>
    <input type="text" id="features" name="features" placeholder="e.g. 200"><br/><br/>
    <input type="submit" value="Predict">
  </form>
  {% if prediction %}
  <h3>Predicted Sales: {{ prediction }}</h3>
  {% endif %}
</body>
</html>
"""


@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Sales Prediction API is running on version 2'})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        features_raw = request.form.get('features')
        if features_raw:
            features_list = [float(x) for x in features_raw.split(',')]
            features = np.array(features_list).reshape(1, -1)
            prediction = float(model.predict(features)[0])
            return render_template_string(FORM_HTML, prediction=prediction)
        else:
            return render_template_string(FORM_HTML, prediction="Invalid input")
    # GET method just shows form without prediction
    return render_template_string(FORM_HTML, prediction=prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
