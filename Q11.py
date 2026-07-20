from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data=load_iris()
X=data.data
y=(data.target == 0).astype(int)

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(X_train,Y_train)

y_pred=model.predict(X_test)

cm=confusion_matrix(Y_test,y_pred)

disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Class 0", "Class 0"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report : ",classification_report(Y_test,y_pred))