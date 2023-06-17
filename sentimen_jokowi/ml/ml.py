from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import pandas as pd

frame = pd.read_csv("jokowi 03 periode - 02/labelin_sentimen_jokowi.csv")

count_vector = CountVectorizer()
x_train_count = count_vector.fit(frame["stemming"].values.astype(str))

tfidf = TfidfTransformer()
x_tf_idf = tfidf.fit_transform(x_train_count)

print(x_tf_idf)
