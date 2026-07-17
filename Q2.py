import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

x=np.random.rand(100,1)*100
y=3*x+np.random.randn(100,1)*2

X_train,X_test,Y_train,Y_test=train_test_split(x,y,train_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x,y)

y_pred=model.predict(X_test)

print("Slope : ",model.coef_[0][0])
print("Intercept : ",model.intercept_[0])

plt.scatter(X_test,Y_test,color="blue",label="Actual")
plt.plot(X_test,y_pred,color="red",label="Predicted")
plt.legend()
plt.xlabel("x")
plt.ylabel("Y")
plt.show()

mse=mean_squared_error(Y_test,y_pred)
r2=r2_score(Y_test,y_pred)

print("MSE : ",mse)
print("R2_Score : ",r2)