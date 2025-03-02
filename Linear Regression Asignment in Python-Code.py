#imporing pandas
import pandas as pd
#importing Sklearn library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data= pd.read_csv('/content/Reg_Assignment _Data Set/train.csv')
data.head()

# Dropped the columns with more than 40% missing values
data_cleaned = data.dropna(thresh=len(data)*0.4,axis=1)
data.head()

#Seperated as Feature and Label
X = data_cleaned[['OverallQual']]
y = data_cleaned[['SalePrice']]
# Split the dataset into features (X) and labels (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f'Coefficient of determination is {r2_score(y_test, y_pred)}')

import matplotlib.pyplot as plt
# Plotting the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='red', label='Test data')
plt.plot(X_test, y_pred, color='green', linewidth=2, label='Predictions')
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')
plt.title('Relationship between OverallQual and SalePrice')
plt.legend()
plt.show()

# Predict the housing price for a new OverallQual value

new_overall_qual = [[11]]
predicted_price = model.predict(new_overall_qual)
print(f'Predicted housing price for OverallQual 11 is {predicted_price[0][0]}')