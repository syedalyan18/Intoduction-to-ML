import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures

data=fetch_california_housing(as_frame=True)
df=data.frame
X=df[['MedInc']]
Y=df[['MedHouseVal']]


poly=PolynomialFeatures(degree=2,include_bias=False)
X_poly=poly.fit_transform(X)

X_train,X_test,Y_train,Y_test=train_test_split(X_poly,Y,train_size=0.2,random_state=42)
 
ridge_model=Ridge(alpha=1)
ridge_model.fit(X_train,Y_train)
ridge_predictions=ridge_model.predict(X_test)

lasso_model=Lasso(alpha=1)
lasso_model.fit(X_train,Y_train)
lasso_predictions=lasso_model.predict(X_test)

ridge_mse=mean_squared_error(Y_test,ridge_predictions)
print("Ridge MSE : ",ridge_mse)

lasso_mse=mean_squared_error(Y_test,lasso_predictions)
print("Lasso MSE : ",lasso_mse)

plt.figure(figsize=(10,6))
plt.scatter(X_test[:,0],Y_test,color="blue",alpha=0.5,label="Actual Data")
plt.scatter(X_test[:,0],ridge_predictions,alpha=0.5,label="Ridge Predictions",color="red")
plt.scatter(X_test[:,0],lasso_predictions,alpha=0.5,label="Lasso Predictions",color="orange")
plt.title("Ridge vs Lasso Regression")
plt.xlabel("Median Income (Transformed)")
plt.ylabel("Median Housing Value in California")
plt.legend()
plt.show()