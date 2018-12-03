from flask import Flask, render_template
from flask import request, redirect
from flask_cors import CORS
import os
import os.path

app = Flask(__name__)
CORS(app)

@app.route("/")
def first_page():
    return render_template("index.html")

# @app.route('/getPrediction', methods=['POST', 'GET'])
# def getPrediction():
# 	if request.method == 'POST':
# 		print("Reaching here")
# 	return render_template("index.html", result="Slight")

@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':
        asd = request.json
        print(asd)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, threaded=True)