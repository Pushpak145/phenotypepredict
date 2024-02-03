import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import  Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression

le = LabelEncoder()

ranks = {}

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

dataset = pd.read_csv('Dataset/wheatNSGC9K_6Xsamplesheet.csv',sep=',')
#dataset = pd.get_dummies(dataset,drop_first=True)
dataset['Sample_ID'] = pd.Series(le.fit_transform(dataset['Sample_ID']))
dataset['Sample_Plate'] = pd.Series(le.fit_transform(dataset['Sample_ID']))
dataset['Sample_Name'] = pd.Series(le.fit_transform(dataset['Sample_Name']))
dataset['Sample_Well'] = pd.Series(le.fit_transform(dataset['Sample_Well']))
dataset['SentrixBarcode_A'] = pd.Series(le.fit_transform(dataset['SentrixBarcode_A']))
dataset['SentrixPosition_A'] = pd.Series(le.fit_transform(dataset['SentrixPosition_A']))
cols = dataset.columns.values.tolist()
names = []
for i in range(1,len(cols)):
        names.append(cols[i])
print(names)
dataset = dataset.fillna(0)
cols = dataset.shape[1]
X = dataset.values[:, 1:cols] 
Y = dataset.values[:, 0]

minmax = MinMaxScaler()

X = minmax.fit_transform(X)

print(X)
print(Y)

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

cls = GaussianProcessRegressor()
cls = LinearRegression()
cls.fit(X, Y)
ranks['BLUP'] = ranking(np.abs(cls.coef_), names)
print(ranks['BLUP'])

cls = GradientBoostingRegressor()
cls.fit(X, Y)
ranks['GB'] = ranking(cls.feature_importances_, names)
print(ranks['GB'])

clf = SVC(C = 1e5, kernel = 'linear')
clf.fit(X, Y)
ranks['SVM'] = ranking(np.abs(clf.coef_.ravel()), names)
print(ranks['SVM'])

rf = RandomForestRegressor()
rf.fit(X, Y)
ranks['RF'] = ranking(rf.feature_importances_, names)
print(ranks['RF'])

ridge = Ridge(alpha = 7)
ridge.fit(X,Y)
ranks['Ridge'] = ranking(np.abs(ridge.coef_), names)
print(ranks['Ridge'])
print("=====")
# Using Lasso
lasso = Lasso(alpha=.05)
lasso.fit(X, Y)
ranks["Lasso"] = ranking(np.abs(lasso.coef_), names)
print(ranks['Lasso'])

r = {}
for name in names:
    r[name] = round(np.mean([ranks[method][name] for method in ranks.keys()]), 2)
 
methods = sorted(ranks.keys())
#ranks["Mean"] = r
#methods.append("Mean")
 
print("\t\t\t%s" % "\t".join(methods))
for name in names:
        temp = name
        if len(temp) < 22:
                size = 22 - len(temp)
                for k in range(0,size):
                        temp+=' '
        print("%s\t%s" % (temp, "\t".join(map(str, [ranks[method][name] for method in methods]))))

def sumAll(columns,names):
        blup = 0
        count = 0
        for name in names:
                blup = blup + np.sum(columns[name])
                count = count + 1
        blup = (blup / 100) * 100
        return blup
blup_avg =  sumAll(ranks['BLUP'],names)
print(blup_avg)
gb_avg = sumAll(ranks['GB'],names)
print(gb_avg)
la_avg = sumAll(ranks['Lasso'],names)
print(la_avg)
rf_avg = sumAll(ranks['RF'],names)
print(rf_avg)
ri_avg = sumAll(ranks['Ridge'],names)
print(ri_avg)
svm_avg = sumAll(ranks['SVM'],names)
print(svm_avg)

'''        
blup = np.sum(ranks['GB'][name]) / len(ranks['BLUP'][name])
print(blup)
blup = np.sum(ranks['SVM'][name]) / len(ranks['BLUP'][name])
print(blup)
blup = np.sum(ranks['RF'][name]) / len(ranks['BLUP'][name])
print(blup)
blup = np.sum(ranks['Ridge'][name]) / len(ranks['BLUP'][name])
print(blup)
blup = np.sum(ranks['Lasso'][name]) / len(ranks['BLUP'][name])
print(blup)
'''
