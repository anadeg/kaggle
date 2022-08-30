"""
Basic EDA
Data analysis using heatmap
Finding correlation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


TEST_FILE = 'test.csv'
TRAIN_FILE = 'train.csv'


train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)
data = pd.concat([train_data, test_data])

print('----- Train Data ----- \n', train_data.head())
print('----- Test Data ----- \n', test_data.head())
print('----- Data ----- \n', data.head())
target = train_data['Survived']
data = data.drop('Survived', axis=1).reset_index(drop=True)
# survived column is our target

# looking for missing data from train and test data
missing_report = data.isna().sum()
print(missing_report)
# there is extremely large number of missing cabin data
# it is better to drop column

correlation = train_data.corr()
# correlation_columns = correlation.columns

fig, ax = plt.subplots()

ax = sns.heatmap(correlation, annot=True, cmap='YlGnBu')
plt.show()
# passenger id has very low correlation to other features and result, so it will be dropped
# the biggest correlations is:
#   fare --- pclass
#   age --- pclass (but there is a lot of missing values)
#   parch --- sibsp
#   survived --- fare

# sex survival analysis
sns.countplot(x=train_data['Sex'], hue=train_data['Survived'])
plt.title('Survival distribution based on Gender')
plt.show()
# it seems like females were more likely to survive

# age survival analysis
ax = sns.kdeplot(x=train_data['Age'], hue=train_data['Survived'])
ax.set(xlabel='Age', title='Distribution of Age based on target variable')
plt.show()
# children and adults (not teens and elder people) have more chance to survive

# i saw idea to join female and children in one value
# children are people from 0 to 8

# pclass survival analysis
sns.countplot(x=train_data['Pclass'], hue=train_data['Survived'])
plt.title('Survival distribution based on Pclass')
plt.show()
# people from 3rd class were less likely to survive

# embarked survival analysis
sns.countplot(x=train_data['Embarked'], hue=train_data['Survived'])
plt.title('Survival distribution based on Embarked')
plt.show()


# let's deal with missing values

# fare:
ax = sns.kdeplot(x=train_data['Fare'])
ax.set(xlabel='Fare', title='Distribution of Fare')
plt.show()
# it's better to use mode (we have just one missing value)

# age:
ax = sns.kdeplot(x=train_data['Age'])
ax.set(xlabel='Age', title='Distribution of Age')
plt.show()
# values are very different, and we have a lot of missing data
# it is bad idea to use median etc.

# embarked:
sns.countplot(x=train_data['Embarked'])
plt.title('Embarked distribution')
plt.show()
# it is better to use most common value










