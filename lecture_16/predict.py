from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
import pickle

with open("model.pickle", "rb") as mod_pick:
    model = pickle.load(mod_pick)

with open("vector.pickle", "rb") as vec_pick:
    vect = pickle.load(vec_pick)


line = "winner you win a lottery. Please collect your prize"

result = model.predict(vect.transform([line]).todense())

print(result)

