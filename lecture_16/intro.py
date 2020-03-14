from flask import Flask, templating, redirect, request

app = Flask("hello")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
import pickle

with open("model.pickle", "rb") as mod_pick:
    model = pickle.load(mod_pick)

with open("vector.pickle", "rb") as vec_pick:
    vect = pickle.load(vec_pick)


@app.route("/")
def say_hello():
    return templating.render_template("form.html", output="")


@app.route("/", methods=["POST"])
def submit_loc():
    line = request.form["line"]
    result = model.predict(vect.transform([line]).todense())
    return templating.render_template("form.html", output=result)



if __name__ == "__main__":
    app.run()
