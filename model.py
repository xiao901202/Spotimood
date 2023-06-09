from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVR
from sklearn import svm
import pandas as pd
import pickle 

sentences = []
scores = []
rows = pd.read_csv('google_train_117k_dropna.csv')
i = 0 
for index, row in rows.iterrows():
    sentences.append(row['content'])
    scores.append(row['score'])
    i+=1
print("訓練了{}筆資料".format(i))
# 將中文句子轉換為詞向量
vector = TfidfVectorizer() 
X = vector.fit_transform(sentences)
y = np.array(scores)
# 切分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 建立 SVM 回歸模型
# model = SVR(kernel='rbf', C=10, gamma=0.45)
print(X_train.shape[0],X_test.shape[0],y_train.shape[0],y_test.shape[0])

#0.66
# model =svm.LinearSVC(C=1, max_iter=10000)

#0.68
model =svm.SVC(kernel='rbf', gamma=0.5, C=10)

#0.54
# model=svm.SVC(kernel='poly', degree=3, gamma='auto', C=1)


# 訓練及儲存模型
model.fit(X_train, y_train)
with open('google117kmodel_poly.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('google117kvectorizer_poly.pkl', 'wb') as f:
    pickle.dump(vector, f)


# 預測測試集的評分
y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)
print(accuracy)

