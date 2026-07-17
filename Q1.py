import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df=pd.read_csv(url)

features=df[['total_bill','size']]
target=df['tip']

print("Features : \n",features.head())
print("Target : \n",target.head())

X_train,X_test,Y_train,Y_test=train_test_split(features,target,train_size=0.2,random_state=42)

print("Training Data Set : \n",X_train.shape)
print("Testing Data Set : \n",X_test.shape)

sns.pairplot(df,x_vars=["total_bill","size"],y_vars="tip",size=0.2,aspect=0.8,kind="scatter")
plt.title("Features vs Target ")
plt.show()