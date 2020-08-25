import pandas as pd
import numpy as np
import sklearn as sk
train_data = pd.read_csv(r'D:\Sambita\hackathon-ideathon\janata hack-indepenceday\train.csv')
print(np.shape(train_data))
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns", None)
#print(train_data[:3].head(10))
test_data = pd.read_csv(r'D:\Sambita\hackathon-ideathon\janata hack-indepenceday\test.csv')
print(np.shape(test_data))
#print(test_data.head(20))
train_data['label'] = train_data['Computer Science'].astype(str) + train_data['Physics'].astype(str) + train_data['Mathematics'].astype(str) +train_data['Statistics'].astype(str) +  train_data['Quantitative Biology'].astype(str) +train_data['Quantitative Finance'].astype(str)

#print(train_data['label'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data['ABSTRACT'], train_data['label'], random_state=1)
#print(y_train)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)
#print(X_train_cv)
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_cv, y_train)
predictions = naive_bayes.predict(X_test_cv)
from sklearn.metrics import accuracy_score, precision_score, recall_score
print('Accuracy score: ', accuracy_score(y_test, predictions))
print('Precision score: ', precision_score(y_test, predictions,average='micro'))
print('Recall score: ', recall_score(y_test, predictions,average='micro'))
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar=False,xticklabels=['yes', 'no'], yticklabels=['yes', 'no'])
plt.xlabel('true label')
plt.ylabel('predicted label')
#predicting test data

cv1 = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
z_test_cv = cv.transform(test_data['ABSTRACT'])
pred_test = naive_bayes.predict(z_test_cv)
#print(pred_test)
type(pred_test)
df = pd.DataFrame(data =pred_test)
test_data['label'] = df
print(test_data)
test_data['Computer Science']=test_data['label'].str[0]
test_data['Physics']=test_data['label'].str[1]
test_data['Mathematics']=test_data['label'].str[2]
test_data['Statistics']=test_data['label'].str[3]
test_data['Quantitative Biology']=test_data['label'].str[4]
test_data['Quantitative Finance']=test_data['label'].str[5]
#print(test_data.head(20))
test_data1= test_data.drop(columns=['label','ABSTRACT','TITLE'])
print(test_data.head(20))
test_data1.to_csv(r'D:\Sambita\hackathon-ideathon\janata hack-indepenceday\submission1.csv',index=False)
