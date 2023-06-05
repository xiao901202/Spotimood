# -*- coding: UTF-8 -*-
import resultFlask as rf

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'hello!!'

@app.route('/predict', methods=['POST'])
def postInput():
    # 取得前端傳過來的數值
    insertSentence = request.get_json()
    input =insertSentence['Sentence']
    result = rf.predict(input)
    print(result)
    return jsonify({'return': str(result)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)