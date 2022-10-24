import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv("Classification/dataset.csv")
df["oldpeak"]=df["oldpeak"].astype('int64')
x=df[df.columns.difference(["output"])]
y=df["output"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train=StandardScaler().fit(x_train).transform(x_train)
x_test=StandardScaler().fit(x_test).transform(x_test)
f, axes = plt.subplots(2, 2)

print("--------------------------- K-Nearest Neighbors(KNN) ------------------------------------")
knn=KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred=knn.predict(x_test)
conf_matrix=confusion_matrix(y_test,y_pred,labels=[0,1])
disp=ConfusionMatrixDisplay(conf_matrix,display_labels=[0,1])
disp.plot(ax=axes[0][0])
disp.ax_.set_title('K-Nearest Neighbors(KNN)')
disp.ax_.set_xlabel('Predicted label')
disp.ax_.set_ylabel('True label')
eval_acc=classification_report(y_test, y_pred)
print(eval_acc)
# ---  Decision Tree ---
print("--------------------------- Decision Tree ------------------------------------")
dec_tree=DecisionTreeClassifier()
dec_tree.fit(x_train, y_train)
y_pred=dec_tree.predict(x_test)
conf_matrix=confusion_matrix(y_test,y_pred,labels=[0,1])
disp=ConfusionMatrixDisplay(conf_matrix,display_labels=[0,1])
disp.plot(ax=axes[0][1])
disp.ax_.set_title('Decision Tree')
disp.ax_.set_xlabel('Predicted label')
disp.ax_.set_ylabel('True label')
eval_acc=classification_report(y_test, y_pred)
print("DecisionTree -> ",eval_acc)
# tree.plot_tree(dec_tree,filled=True)
# plt.show()
# -- Logistic Regression ---
print("--------------------------- Logistic Regression ------------------------------------")
log_reg=LogisticRegression()
log_reg.fit(x_train, y_train)
y_pred=log_reg.predict(x_test)
conf_matrix=confusion_matrix(y_test,y_pred,labels=[0,1])
disp=ConfusionMatrixDisplay(conf_matrix,display_labels=[0,1])
disp.plot(ax=axes[1][0])
disp.ax_.set_title('Logistic Regression')
disp.ax_.set_xlabel('Predicted label')
disp.ax_.set_ylabel('True label')
eval_acc=classification_report(y_test, y_pred)
print(eval_acc)
# -- SVM ---
print("--------------------------- Support Vector Machine(SVM) ------------------------------------")
svm=SVC()
svm.fit(x_train, y_train)
y_pred=svm.predict(x_test)
conf_matrix=confusion_matrix(y_test,y_pred,labels=[0,1])
disp=ConfusionMatrixDisplay(conf_matrix,display_labels=[0,1])
disp.plot(ax=axes[1][1])
disp.ax_.set_title('Support Vector Machine(SVM)')
disp.ax_.set_xlabel('Predicted label')
disp.ax_.set_ylabel('True label')
eval_acc=classification_report(y_test, y_pred)
print(eval_acc)

# -- Show --
plt.subplots_adjust(wspace=0.8, hspace=0.8)
plt.show()
