import pickle
from ckiptagger import WS, POS
ws = WS("./data")
pos = POS("./data")

with open(r'google72kmodel_poly.pkl', 'rb') as f:
    model = pickle.load(f)
with open(r'google72kvectorizer_poly.pkl', 'rb') as f:
    vector = pickle.load(f)

def predict(sentence):
    sentences = []
    words = ws([sentence])
    content = '|'.join(words[0])
    sentences.append(content)
    X = vector.transform(sentences)
    pred=model.predict(X)
    return pred[0]
