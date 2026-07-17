import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(42)
x=np.random.rand(100,1)*10
y=3*x**2 + 2*x + np.random.randn(100,1)*5

poly_features=PolynomialFeatures(degree=2,include_bias=False)
x_poly=poly_features.fit_transform(x)

X_train,X_test,Y_train,Y_test=train_test_split(x_poly,y,train_size=0.2,random_state=42)
 
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