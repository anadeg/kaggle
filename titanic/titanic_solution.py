import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer


TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
ANSWER_FILE = 'answer.csv'


def convert_table(table):
    # drop useless features
    table.drop(columns=['PassengerId', 'Cabin', 'Ticket'], inplace=True)

    # combine family dependent values
    table['Family'] = np.where(table['SibSp'] + table['Parch'] > 0, 1, 0)
    table.drop(['SibSp', 'Parch'], axis=1, inplace=True)

    # extract names titles
    table['Title'] = table['Name'].str.extract(r'([A-Za-z]+)\.', expand=True)
    mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
               'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
    table['Title'].replace(mapping, inplace=True)
    table.drop('Name', axis=1, inplace=True)

    # change strings to integers
    # i.e. 'female'/'male' ===> 0/1
    le = LabelEncoder()
    table['Title'] = le.fit_transform(table['Title'])
    table['Embarked'] = le.fit_transform(table['Embarked'])
    table['Sex'] = le.fit_transform(table['Sex'])

    # impute missing values
    knni = KNNImputer()
    table[['Age', 'Fare', 'Embarked']] = knni.fit_transform(table[['Age', 'Fare', 'Embarked']])

    # age and fare have big scale ===> change the range
    table['FareCode'] = pd.qcut(table['Fare'], 5, labels=False)
    table['AgeCode'] = pd.qcut(table['Age'], 5, labels=False)
    table.drop(columns=['Fare', 'Age'], inplace=True)

    sc = StandardScaler()
    table = sc.fit_transform(table)
    return table


train_data = pd.read_csv(TRAIN_FILE)
X_train = train_data.drop(['Survived'], axis=1)
Y_train = train_data['Survived']

X_train = convert_table(X_train)

test_data = pd.read_csv(TEST_FILE)
test_answers = pd.read_csv(ANSWER_FILE)

ids = test_data['PassengerId']

X_test = test_data
Y_test = test_answers['Survived']
X_test = convert_table(X_test)

# trying out different algorithms

xgbc = XGBClassifier()
xgbc.fit(X_train, Y_train)
# prediction = xgbc.predict(X_test).astype(int)
print('XGBoost Score --- ', xgbc.score(X_test, Y_test))

svc = SVC()
svc.fit(X_train, Y_train)
prediction = svc.predict(X_test).astype(int)
print('SVC Score --- ', svc.score(X_test, Y_test))

lr = LogisticRegression()
lr.fit(X_train, Y_train)
# prediction = lr.predict(X_test)
print('Logistic Regression Score --- ', lr.score(X_test, Y_test))

rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
# prediction = rfc.predict(X_test)
print('Random Forest Score --- ', rfc.score(X_test, Y_test))

knnc = KNeighborsClassifier()
knnc.fit(X_train, Y_train)
print('KNN Score --- ', knnc.score(X_test, Y_test))

# data goes from this one
cbc = CatBoostClassifier(verbose=0)
cbc.fit(X_train, Y_train)
# prediction = cbc.predict(X_test)
print('Cat Boost Score --- ', cbc.score(X_test, Y_test))

# writing result in csv file
output = pd.DataFrame({'PassengerId': ids, 'Survived': prediction})
output.to_csv('submission.csv', index=False)





