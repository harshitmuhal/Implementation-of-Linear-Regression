import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

dataset = datasets.load_boston()
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
           'TAX', 'PTRATIO', 'B', 'LSTAT']
df = pd.DataFrame(dataset.data, columns=columns)
y = dataset.target
df['target'] = y
print(df.isnull())
# to check if there are any missing values in the data

corr = df.corr().round(2)
# to round off correlation coefficients upto 2 digits

sns.heatmap(data=corr, annot=True)
# A heatmap is a two-dimensional graphical representation of data where the individual values that are contained in a matrix are represented as colors.
# annot has to be true to print values inside box, otherwise only colour will be shown

plt.show()
features = ['LSTAT', 'RM', 'PTRATIO']
for i, col in enumerate(features):
    plt.subplot(1, len(features), i + 1)
    x = df[col]
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('target variable')
plt.show()
data = pd.DataFrame(np.c_[df['LSTAT'], df['RM'], df['PTRATIO']], columns=features)

# using train/test split for evaluation
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.3)
m_train = x_train.shape[0]
m_test = x_test.shape[0]
lm = LinearRegression()
lm.fit(x_train, y_train)
y_pred = lm.predict(x_test)
plt.scatter(y_test, y_pred)
plt.show()

# for a fit model y_test,y_pred should be linear

mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
print('mean squared error : ', mse, ' root mean squared error : ', rmse)
sample = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
           6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
           1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]

# taking only LMSTAT,PTRATIO,RM

sample_ = [[1.09700000e+01, 6.32600000e+00, 1.86000000e+01]]
prediction = lm.predict(sample_)
print(prediction)
