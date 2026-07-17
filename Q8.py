import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
from sklearn.model_selection import train_test_split

np.random.seed(42)
n_samples=200
X=np.random.rand(n_samples,2)*10
Y= (X[:,0]*1.5 + X[:,0] > 1).astype(int)

df=pd.DataFrame(X,columns=['Age', 'Salary'])
df['Purchase']=Y

X_train,X_test,Y_train,Y_test=train_test_split(df[['Age','Salary']], df['Purchase'],train_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(X_train,Y_train)

y_pred=model.predict(X_test)

print("Accuracy : ",accuracy_score(Y_test,y_pred))
print("Precision : ",precision_score(Y_test,y_pred))
print("Recall : ",recall_score(Y_test,y_pred))
print("F1 : ",f1_score(Y_test,y_pred))
print("\nClassification Report : ",classification_report(Y_test,y_pred))

import matplotlib.pyplot as plt

X_min,X_max=df["Age"].min()-1,df["Age"].max()+1
Y_min,Y_max=df["Salary"].min()-1, df["Salary"].max()+1

xx,yy=np.meshgrid(np.arange(X_min,X_max,0.1),np.arange(Y_min,Y_max,0.1))

z=model.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)

plt.contourf(xx,yy,z,alpha=0.8,cmap="coolwarm")
plt.scatter(X_test["Age"],X_test["Salary"],c=Y_test,edgecolor="k",cmap="coolwarm")
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()