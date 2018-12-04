from flask import Flask, render_template
from flask import request, redirect, make_response
# from flask.ext.responses import json_response
from flask_cors import CORS
import os
import os.path
import json
from preprocessNew import prediction

app = Flask(__name__, static_url_path = "", static_folder = "tmp")
CORS(app)

#initialize the data and model
obj = prediction()

@app.route("/")
def first_page():
    return render_template("index.html")
    
@app.route('/predict', methods=['POST'])
def predict():
     data = request.get_json()
     print(data)
     val = obj.predictResult(data)
     r = {"result":val}
     # result = "Slight"
     resp = make_response(json.dumps(r))
     resp.status_code = 200
     resp.headers['Access-Control-Allow-Origin'] = '*'
     return resp
     
     #return json_response(r, status_code=201)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, threaded=True)