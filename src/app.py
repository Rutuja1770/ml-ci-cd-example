from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Simple ML model (y = 2x + 3)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([5, 7, 9, 11, 13])
model = LinearRegression().fit(X, y)

@app.route("/")
def home():
    return "ML Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    x_value = np.array([[data["x"]]])
    prediction = model.predict(x_value)[0]
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
