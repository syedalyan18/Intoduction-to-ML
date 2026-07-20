from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data=fetch_california_housing()
X,y=data.data,data.target 

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,Y_train)

y_pred=model.predict(X_test)

mse=mean_squared_error(Y_test,y_pred)
mae=mean_absolute_error(Y_test,y_pred)
r2=r2_score(Y_test,y_pred)

print(f"Mean Absolute Error : ,{mae:.2f}")
print(f"Mean Squared Error : ,{mse:.2f}")
print(f"R2 Score : ,{r2:.2f}")