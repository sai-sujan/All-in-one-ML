import evalml
import pandas as pd

df=pd.read_csv('loan.csv')
X=df.drop(['Id','SalePrice'],axis=1)
y = df['SalePrice']
X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(X, y, problem_type='Regression')
from evalml.automl import AutoMLSearch
automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='Regression')
automl.search()
print(automl.rankings)
