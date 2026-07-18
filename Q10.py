from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data=load_iris()
X,y=data.data, data.target

X_train,X_test,Y_train,Y_test=train_test_split(X,y,train_size=0.2,random_state=42)

model=LogisticRegression(max_iter=200)
model.fit(X_train,Y_train)

y_pred=model.predict(X_test)

cm=confusion_matrix(Y_test,y_pred)

disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=data.target_names)
disp.plot(cmap="Blues")
plt.show()

print("Classification Report : ",classification_report(Y_test,y_pred))