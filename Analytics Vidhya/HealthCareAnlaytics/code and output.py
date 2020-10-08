#Code 1 With dummy encoding and logistic regression 

import pandas as pd 

train = pd.read_csv(r'../input/av-janatahack-healthcare-hackathon-ii/Data/train.csv')
test =pd.read_csv(r'../input/av-janatahack-healthcare-hackathon-ii/Data/test.csv')
print(train['Stay'].value_counts())
print(train.columns)

#correlation between columns
features = train[['Ward_Type','Department','Bed Grade','Available Extra Rooms in Hospital','Type of Admission','Admission_Deposit','Severity of Illness','Visitors with Patient','Age','Stay']]

pearson_correl =features.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson')
#print(pearson_correl)
features.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson')
#chisquare test
from scipy.stats import chi2_contingency
import numpy as np
fac_pair = [(i,j) for i in features.columns.values for j in features.columns.values]
chi2 , p_value = [],[]
for f in fac_pair:
    if f[0] != f[1]:
        chitest = chi2_contingency(pd.crosstab(features[f[0]],features[f[1]]))
        chi2.append(chitest[0])
        p_value.append(chitest[1])
    else:
        chi2.append(0)
        p_value.append(0)

chi2_2 =np.array(chi2).reshape(10,10)
pd.DataFrame(chi2_2,index=features.columns.values,columns=features.columns.values) 
#print(chi3)


from sklearn.model_selection import train_test_split

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
feature1=['Ward_Type','Type of Admission','Available Extra Rooms in Hospital','Visitors with Patient']
train_hot_encode = pd.get_dummies(train[feature1])
X_train, X_test, y_train,y_test = train_test_split(train_hot_encode,train['Stay'],test_size =0.20, shuffle =True)


model = LogisticRegression()
model.fit(X_train,y_train)
pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score
print('accuracy score is {:4f}'.format(accuracy_score(y_test,pred)))
print('Precision score: ', precision_score(y_test, pred,average='micro'))
print('Recall score: ', recall_score(y_test, pred,average='micro'))
#print('f1 score is {:4f}'.format(f1_score(y_test,pred)))


test_hot_encode = pd.get_dummies(test[feature1])
pred_test =model.predict(test_hot_encode)
test_new1 =pd.DataFrame()
test_new1['case_id'] =test['case_id']
test_new1['Stay'] =  pd.DataFrame(pred_test)
test_new1.head(20)
test_new1.to_csv(r'D:\Sambita\hackathon-ideathon\Analytics Vidhya\janata hack-health care\Submission1.csv')
'''
#Output

21-30                 87491
11-20                 78139
31-40                 55159
51-60                 35018
0-10                  23604
41-50                 11743
71-80                 10254
More than 100 Days     6683
81-90                  4838
91-100                 2765
61-70                  2744
Name: Stay, dtype: int64
Index(['case_id', 'Hospital_code', 'Hospital_type_code', 'City_Code_Hospital',
       'Hospital_region_code', 'Available Extra Rooms in Hospital',
       'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade',
       'patientid', 'City_Code_Patient', 'Type of Admission',
       'Severity of Illness', 'Visitors with Patient', 'Age',
       'Admission_Deposit', 'Stay'],
      dtype='object')
accuracy score is 0.359550
Precision score:  0.35954967968848134
Recall score:  0.35954967968848134
'''

# Code 2 with hashing encoder method and logistic regression & lgbm
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from category_encoders.hashing import HashingEncoder
X_train1, X_test1, y_train1,y_test1 = train_test_split(train[feature1],train['Stay'],test_size =0.20, shuffle =True)
he = HashingEncoder(cols=['Ward_Type','Type of Admission','Available Extra Rooms in Hospital','Visitors with Patient']).fit(X_train1, y_train)
data = he.transform(X_train1)
data_test = he.transform(X_test1)
print(data.head(20))
'''
#output

        col_0  col_1  col_2  col_3  col_4  col_5  col_6  col_7
225917      1      0      0      0      2      0      1      0
204389      0      0      0      2      1      0      1      0
60523       0      0      0      1      1      1      1      0
32187       0      0      0      1      2      0      1      0
103972      0      0      0      1      2      0      1      0
211224      1      0      0      0      2      0      1      0
88155       0      0      0      3      0      0      1      0
104466      0      0      0      1      2      0      1      0
135541      1      0      0      0      1      1      1      0
315730      0      0      0      2      1      0      1      0
286037      1      0      0      0      2      0      1      0
29021       0      0      0      1      2      0      1      0
93904       0      0      0      3      0      0      1      0
13460       0      0      0      1      2      0      1      0
115808      0      0      0      1      2      0      1      0
158301      1      0      0      1      1      0      1      0
98860       0      0      0      1      2      0      1      0
39113       1      0      0      1      1      0      1      0
34219       1      0      0      1      1      0      1      0
55039       1      0      0      0      2      0      1      0

'''
model = LogisticRegression()
model.fit(data,y_train1)
pred1 = model.predict(data_test)
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score
print('accuracy score is {:4f}'.format(accuracy_score(y_test,pred1)))
print('Precision score: ', precision_score(y_test, pred1,average='micro'))
print('Recall score: ', recall_score(y_test, pred1,average='micro'))

test_hot_encode1 = he.transform(test[feature1])
pred_test =model.predict(test_hot_encode1)
test_new1 =pd.DataFrame()
test_new1['case_id'] =test['case_id']
test_new1['Stay'] =  pd.DataFrame(pred_test)
print(test_new1.head(20))
test_new1.to_csv(r'D:\Sambita\hackathon-ideathon\Analytics Vidhya\janata hack-health care\Submission1_2.csv')
'''
accuracy score is 0.260583
Precision score:  0.260582841351589
Recall score:  0.260582841351589
'''
train_hot_encode = pd.get_dummies(train.drop(columns = ['Stay']))
test_hot_encode = pd.get_dummies(test)
print(test_hot_encode.shape)
print(train_hot_encode.shape)
X_train2, X_test2, y_train2,y_test2 = train_test_split(train_hot_encode,train['Stay'],test_size =0.20, shuffle =True)
import lightgbm as lgb
params = {}
params['learning_rate'] = 0.01
params['max_depth'] = 18
params['n_estimators'] = 3000
params['objective'] = 'multiclass'
params['boosting_type'] = 'gbdt'
params['subsample'] = 0.7
params['random_state'] = 42
params['colsample_bytree']=0.7
params['min_data_in_leaf'] = 55
params['reg_alpha'] = 1.7
params['reg_lambda'] = 1.11
#params['class_weight'] = {'0':.10,'1':.12,'2':0.13,'3':0.11,'4':0.09,'5':0.10,'6':0.09,'7':0.09,'8':0.09,'9':0.09,'10':0.09}
#params['class_weight']: {'A': 0.20, 'B': 0.20, 'C': 0.25,'D': 0.30}


clf = lgb.LGBMClassifier(**params)
clf.fit(X_train2, y_train2)
lgbm_pred=clf.predict(X_test2)
lgbm_acc=accuracy_score(y_test2,lgbm_pred)
print('accuracy_score : 'lgbm_acc)
lgbm_pred1 = clf.predict(test_hot_encode)
test_new1['case_id'] =test['case_id']
test_new1['Stay'] =  pd.DataFrame(lgbm_pred1)
print(test_new1.head(20))
test_new1.to_csv(r'D:\Sambita\hackathon-ideathon\Analytics Vidhya\janata hack-health care\Submission1_4.csv', index = False)
'''
0.4259986182640372
    case_id   Stay
0    318439   0-10
1    318440  51-60
2    318441  21-30
3    318442  21-30
4    318443  51-60
5    318444  21-30
6    318445  21-30
7    318446  11-20
8    318447  21-30
9    318448  21-30
10   318449  21-30
11   318450  31-40
12   318451  21-30
13   318452  21-30
14   318453  31-40
15   318454  31-40
16   318455  11-20
17   318456  21-30
18   318457  21-30
19   318458  21-30 '''