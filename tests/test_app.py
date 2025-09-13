import json
from src.app import app

def test_home():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"ML Model API is running!" in response.data

def test_predict():
    client = app.test_client()
    response = client.post("/predict", 
                           data=json.dumps({"x": 10}),
                           content_type="application/json")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "prediction" in data
