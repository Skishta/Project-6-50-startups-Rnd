'''بسم الله الرحمن الرحيم'''               

                            #Project 6 (Dataset contains data about 50 startups their spend in R&D, administration, marketing spend, state of the headquarters) using sheet excel of real employees
#
##ANOTHER PROJECT FOR Multi-Linear REGRESSION
#Import all liberaries (all liberaries explained before)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
data=pd.read_csv("50_Startups.csv")
le=preprocessing.LabelEncoder()
data["State"]=le.fit_transform(data["State"])
X=data.iloc[:,2:-2]
Y=data["Profit"]
X_train,X_test,Y_train,Y_test=train_test_split(xpoly,Y,test_size=0.2,random_state=0)
reg=LinearRegression()
reg.fit(X_train,Y_train)
ypred=reg.predict(X_test)
print(ypred)
print(Y_test)
poly_obj=PolynomialFeatures(degree=6)
xpoly=poly_obj.fit_transform(X)
x=data.iloc[:,2:-2]
plt.scatter(X,Y)
plt.plot(xpoly,reg.predict(xpoly))
plt.show()