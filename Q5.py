import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures

data=fetch_california_housing(as_frame=True)
df=data.frame
X=df[['MedInc']]
Y=df[['MedHouseVal']]


poly=PolynomialFeatures(degree=2,include_bias=False)
X_poly=poly.fit_transform(X)

model=LinearRegression()
model.fit(X_poly,Y)

y_pred=model.predict(X_poly)

plt.figure(figsize=(10,6))
plt.scatter(X,Y,color="blue",label="Actual Curve",alpha=0.2)
plt.scatter(X,y_pred,label="Predicted Curve",color="red",alpha=0.2)
plt.title("Polynomial Regression")
plt.xlabel("Median Income in California")
plt.ylabel("Median Housing Value in California")
plt.legend()
plt.show()

mse=mean_squared_error(Y,y_pred)
print("Mean Squared Error : ",mse)