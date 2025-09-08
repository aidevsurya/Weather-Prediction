from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from pandas import read_csv as read
from pickle import dump

data = read("seattle-weather.csv")

X = data.iloc[:,0:-1]
Y = data.iloc[:,-1]
X = X.drop(['date'],axis=1).values

LE = LabelEncoder()
Yenc = LE.fit_transform(Y)

Xtr,Xte,Ytr,Yte = train_test_split(X,Yenc,test_size=0.2)
classs = LE.classes_
# print(classs)
RFC = RandomForestClassifier(criterion="entropy")
RFC.fit(Xtr,Ytr)
file = open("Weather_model.model","wb")
dump(RFC, file)
ans = RFC.predict(Xte)
# for i,j in zip(Yte,ans):
#     print(classs[i],classs[j],j)
# print(MNB.score(Xte,Yte))

# 0.0, 8.3, 2.8, 4.1
Precipitation = 0.0
MaxTemp = 8.3
MinTemp = 2.8
Wind = 4.1

prediction = RFC.predict([[Precipitation,MaxTemp,MinTemp,Wind]])

print(f"""
Precipatation = {Precipitation}
Max Temp = {MaxTemp}
Min Temp = {MinTemp}
Wind = {Wind}
Weather Prediction = {classs[prediction]}
Accuracy Score = {RFC.score(Xte,Yte)}
""",)
