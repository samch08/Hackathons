import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

titan_df=pd.read_csv("../input/train.csv")
test_titan_df=pd.read_csv("../input/test.csv")
titan_df.drop(['PassengerId','Name'],1,inplace=True)
test_titan_df.drop(['Name'],1,inplace=True)
#ways of exploring data
#print(titan_df.head())
titan_df.describe()
print(titan_df.columns)
#converting columns to numeric 
#titan_df.convert_objects(convert_numeric=True)
titan_df.fillna(0,inplace=True)
test_titan_df.fillna(0,inplace=True)
titan_df1=pd.get_dummies(titan_df,columns=['Sex','Embarked'],drop_first=True)
print(titan_df1.columns)
test_df1=pd.get_dummies(test_titan_df,columns=['Sex','Embarked'],drop_first=True)
test_df1

# Any results you write to the current directory are saved as output.

# %% [code]
#Visualize data
from matplotlib import pyplot as plt
x=titan_df.Sex
y=titan_df.Survived
for x in titan_df:
    if x ==1:
        plt.bar(x,y,color='green',linewidth=5)
        plt.show()




# %% [code]
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
train_cols=['Pclass', 'Sex_male', 'Age', 'SibSp','Parch','Fare','Embarked_Q','Embarked_S']
x_train= titan_df1[train_cols]
y_train=titan_df1.Survived
test_cols=['Pclass', 'Sex_male', 'Age', 'SibSp','Parch','Fare','Embarked_Q','Embarked_S']
x_test=test_df1[test_cols]
#class_titan=DecisionTreeClassifier(random_state=0)
class_titan=RandomForestClassifier(n_estimators=100,oob_score=True)
ct=class_titan.fit(x_train, y_train)
y_test=class_titan.predict(x_test)
# Extract single tree
estimator = class_titan.estimators_[5]
print(y_test.shape)
print(y_train.shape)
print(classification_report(y_train.head(418),y_test))
accuracy=accuracy_score(y_test,y_train.tail(418))
print("Accuracy is %0f" %accuracy)
class_titan.score(x_test,y_test)
#print(f'Out-of-bag score estimate: {class_titan.oob_score_:.3}')

# %% [code]
# Tree generation

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = test_cols,
                #class_names = y_train,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.jpg', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.jpg')

# %% [code]
#taking passenger ids from test data

passids = test_titan_df['PassengerId']
out=pd.DataFrame({'PassengerId': passids, 'Survived': y_test})
print(out)
