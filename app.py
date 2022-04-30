from io import BytesIO
from flask import Flask, request, jsonify, send_file
import numpy as np
import pickle
app = Flask(__name__)

@app.before_first_request
def setup():
    # run setup stuff
    return

@app.route("/")
def index():
    return "<h1>Home Page</h1>"

@app.route("/api/fetch-model/<model_name>", methods=['POST'])
def fetch_model(model_name):
    if model_name == 'model1':
        return jsonify({'model': 'model1'})
    elif model_name == 'model2':
        return jsonify({'model': 'model2'})
    elif model_name == 'model3':
        return jsonify({'model': 'model3'})
    elif model_name == 'model4':
        return jsonify({'model': 'model4'})
    elif model_name == 'model5':
        return jsonify({'model': 'model5'})
    else:
        return jsonify({"error": "error"})
    
@app.route("/api/fetch-user/<id_number>", methods=['POST'])
def fetch_single_user(id_number):
    return jsonify({"User number": id_number})

@app.route("/api/fetch-users/<start_id>-<end_id>", methods=['POST'])
def fetch_multiple_user(start_id, end_id):
    try:
        start = int(start_id)
        end = int(end_id)
        if start > 0 and start < 106 and end > start and end < 107:
            return jsonify({"A": start, "B": end})
        else:
            return jsonify({"error": "Bad input"}), 400
    except ValueError:
        return jsonify({"error": "Bad input"}), 400
        
@app.route("/api/fetch-attach/<attack_id>", methods=['POST'])
def fetch_attack(attack_id):
    if attack_id == "GAN":
        return send_file("")
    return "Test"

@app.route("/api/test", methods=['POST', "GET"])
def test():
    filename = "test"
    testBytes = pickle.dumps({"Testkey": ["Value1, Value2"]})
    buffer = BytesIO()
    buffer.write(testBytes)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, mimetype="text/plain", attachment_filename="test.pkl")

if __name__ == '__main__':
    app.run(Debug=True)