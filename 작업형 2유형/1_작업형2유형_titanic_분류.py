import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score

train=pd.read_csv('./titanic_train.csv')
test=pd.read_csv('./titanic_test.csv')

print(train.shape,test.shape)

'''
PassengerId : 각 승객의 고유 번호
Survived : 생존 여부 -> 0 = 사망 / 1 = 생존
Pclass : 객실 등급
Name : 이름
Sex : 성별
Age : 나이
SibSp : 동반한 형제자매와 배우자 수
Parch : 동반한 부모, 자식 수
Ticket : 티켓의 고유넘부
Fare : 티켓의 요금
Cabin : 객실 번호
Embarked : 승선한 항

'''
train.head()

# PassengerId를 비롯해 모델을 돌리는 작업에 필요없는 컬럼 삭제하기
x_train = train.drop(columns = ['PassengerId','Survived','Name','Ticket','Cabin'])

x_test_id = test['PassengerId'] # 결과표에 붙일 예정
x_test = test.drop(columns = ['PassengerId','Name','Ticket','Cabin'])

# train에 있는 종속변수 survived y_train에 저장하기
y_train = train[['Survived']]

x_train.isnull().sum()

x_test.isnull().sum()

x_train['Age'] = x_train['Age'].fillna(x_train['Age'].median())
x_test['Age'] = x_test['Age'].fillna(x_test['Age'].median())

'''
method의 종류로는 결측치 바로 뒤에값으로 채워주는'bfill'과 앞에 값으로 채워주는 'ffill'이 있다.
method를 적용하고나면 꼭 isnull().sum()으로 확인해야한다.
왜냐하면 맨 앞과 맨 마지막 컬럼이 nan값이면 채워질 값이 없어 두개의 method중 하나는 여전히 nan값으로 남는다.
위와같은 이유로 해당 csv파일에서 method를 'ffill'을 적용시킨다면 결측치가 다 채워지지 않는다.
'''

nan = ['Embarked','Fare']

for i in nan:
    x_train[i] = x_train[i].fillna(method = 'bfill') # 결측치를 바로 뒤에 값으로 채운다
    x_test[i] = x_test[i].fillna(method='bfill')

# 확인
x_train.isnull().sum()
x_test.isnull().sum()

# label encoding작업 전 dytpe이 object , category인 즉, 숫자가 아닌 문자열인 컬럼을 찾는다.
x_train.info()
x_test.info()

# LabelEncoder 작업
encoder = LabelEncoder()
label = ['Sex','Embarked']

x_train[label] = x_train[label].apply(encoder.fit_transform)
x_test[label] = x_test[label].apply(encoder.fit_transform)

# 확인하기
x_train.head()

x_train.describe() # min, max 값들을 기준으로 연속형 값들을 minmaxscaling 한다.

# MinMaxScaling 작업
scaler = MinMaxScaler()
minmax = ['Age','Fare']

scaler.fit(x_train[minmax])
x_train[minmax] = scaler.transform(x_train[minmax])

scaler.fit(x_test[minmax])
x_test[minmax] = scaler.transform(x_test[minmax])

# train, validation 나누기
x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, stratify = y_train['Survived'],
                                                   test_size = 0.2, random_state = 2022)

# RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100 ,max_depth = 3)
rfc.fit(x_tr, y_tr)
pred = rfc.predict_proba(x_val)[:,1] # predict_proba를 한다면 뒤에 [:,1]을 해줘야됨

print(roc_auc_score(y_val, pred)) # 1에 가까울수록 좋음

# XGboost
xg = XGBClassifier(n_estimators = 100 ,max_depth = 3)
xg.fit(x_tr, y_tr)
pred = xg.predict_proba(x_val)[:,1]

print(roc_auc_score(y_val, pred)) # 1에 가까울수록 좋음

# roc_auc_sore값이 xg가 더 높게 나와서 xgboost로 모델 돌림
xg = XGBClassifier(n_estimators = 100 ,max_depth = 3)
xg.fit(x_train, y_train)
pred = xg.predict_proba(x_test)[:,1]

result_pred = pd.DataFrame(pred)
result_pred.columns = ['predict']

result = pd.concat([x_test_id, result_pred],axis=1)

# result.to_csv('수험번호.csv', index=False) #index=False **필수**