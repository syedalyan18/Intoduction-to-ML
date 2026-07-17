import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(42)
x=np.random.rand(100,1)*10
y=3*x**2 + 2*x + np.random.randn(100,1)*5

poly_features=PolynomialFeatures(degree=2,include_bias=False)
x_poly=poly_features.fit_transform(x)

model=LinearRegression()
model.fit(x_poly,y)
y_pred=model.predict(x_poly)

plt.scatter(x,y,color="blue",label="Actual Data")
plt.scatter(x,y_pred,color="red",label="Predicted Data")
plt.title("Polynomial Regression")
plt.legend()
plt.xlabel("x")
plt.ylabel("Y")
plt.show()

mse=mean_squared_error(y,y_pred)
print("Mean Squared Error : ",mse)
