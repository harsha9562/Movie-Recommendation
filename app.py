from flask import Flask
from flask import request
from flask import render_template
import json
from knn_recommender import KNN_recommend

app = Flask(__name__)

@app.route("/")
def hello_world():
    movies = []
    return render_template('index.html', movies=movies)


@app.route("/", methods=['POST'])
def getMovies():
    name = request.form['movie']
    if name is None:
        return json.dumps({"error": "No name provided"}), 400
    try:
        output = KNN_recommend(name)
    except Exception as e:
        print("GOT AN ERROR IN KNN , ERROR -- ", e)
        x = {"status": "erorr", "message": str(e)}
        k = json.dumps(x)
        x = {"status": "erorr", "message": str(e)}
        k = json.dumps(x)
        return k, 500

    # return output as json
    output = json.loads(output)
    print(output['movie'])
    # return output, 200, {'Content-Type': 'application/json'}
    return render_template('index.html', movies=output['movie'])
