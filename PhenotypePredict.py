from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
import seaborn as sns
import os

le = LabelEncoder()

main = tkinter.Tk()
main.title("Machine-Learning for Predicting Phenotype")
main.geometry("1300x1200")

global filename
global X,Y
global ranks
global names
global blup_avg, gb_avg, la_avg, rf_avg, ri_avg, svm_avg

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

def calculateAverage(columns,names):
    value = 0
    count = 0
    for name in names:
        value = value + np.sum(columns[name])
        count = count + 1
    value = (value / 100) * 100
    return value
    
def uploadDataset():
    global filename
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def processDataset():
    global X,Y
    global names
    text.delete('1.0', END)
    sep = ''
    column_index = 1
    y_index = 0
    if os.path.basename(filename) == 'yeast.txt':
        sep = '\t'
        column_index = 1
        y_index = 0
    if os.path.basename(filename) == 'wheatNSGC9K_6Xsamplesheet.csv':
        sep = ','
        column_index = 1
        y_index = 0
    if os.path.basename(filename) == 'rice.csv':
        sep = ','
        column_index = 7
        y_index = 3
    dataset = pd.read_csv(filename,sep=sep)
    dataset = dataset.fillna(0)
    if os.path.basename(filename) == 'yeast.txt':
        dataset['names'] = pd.Series(le.fit_transform(dataset['names']))
    if os.path.basename(filename) == 'wheatNSGC9K_6Xsamplesheet.csv':
        dataset['Sample_ID'] = pd.Series(le.fit_transform(dataset['Sample_ID']))
        dataset['Sample_Plate'] = pd.Series(le.fit_transform(dataset['Sample_ID']))
        dataset['Sample_Name'] = pd.Series(le.fit_transform(dataset['Sample_Name']))
        dataset['Sample_Well'] = pd.Series(le.fit_transform(dataset['Sample_Well']))
        dataset['SentrixBarcode_A'] = pd.Series(le.fit_transform(dataset['SentrixBarcode_A']))
        dataset['SentrixPosition_A'] = pd.Series(le.fit_transform(dataset['SentrixPosition_A']))
    if os.path.basename(filename) == 'rice.csv':
        dataset['NAME'] = pd.Series(le.fit_transform(dataset['NAME']))
        #dataset['CROPYEAR'] = pd.Series(le.fit_transform(dataset['CROPYEAR']))  
    cols = dataset.columns.values.tolist()
    names = []
    for i in range(column_index,len(cols)):
        names.append(cols[i])
    text.insert(END,'Columns in dataset \n\n')
    text.insert(END,names)
    cols = dataset.shape[1]
    X = dataset.values[:, column_index:cols] 
    Y = dataset.values[:, y_index]
    print(Y)
    Y = Y.astype('int')
    minmax = MinMaxScaler()
    X = minmax.fit_transform(X)
    text.insert(END,'\n\n\nTotal records in dataset : '+str(len(X)))

def runAlgorithms():
    global blup_avg, gb_avg, la_avg, rf_avg, ri_avg, svm_avg
    text.delete('1.0', END)
    global ranks
    ranks = {}
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

    lasso = Lasso(alpha=.05)
    lasso.fit(X, Y)
    ranks["Lasso"] = ranking(np.abs(lasso.coef_), names)
    print(ranks['Lasso'])

    r = {}
    for name in names:
        r[name] = round(np.mean([ranks[method][name] for method in ranks.keys()]), 2)
    methods = sorted(ranks.keys())

    text.insert(END,"\t\t\t%s" % "\t".join(methods))
    text.insert(END,"\n")
    print("\t\t\t%s" % "\t".join(methods))
    for name in names:
        temp = name
        if len(temp) < 22:
                size = 22 - len(temp)
                for k in range(0,size):
                        temp+=' '
        text.insert(END,"%s\t%s\n" % (temp, "\t".join(map(str, [ranks[method][name] for method in methods]))))
        print("%s\t%s" % (temp, "\t".join(map(str, [ranks[method][name] for method in methods]))))


    blup_avg =  calculateAverage(ranks['BLUP'],names)
    gb_avg = calculateAverage(ranks['GB'],names)  
    la_avg = calculateAverage(ranks['Lasso'],names)
    rf_avg = calculateAverage(ranks['RF'],names)
    ri_avg = calculateAverage(ranks['Ridge'],names)
    svm_avg = calculateAverage(ranks['SVM'],names)
    gb_avg = gb_avg

def rankingGraph():
    text.delete('1.0', END)
    text.insert(END,'Average BLUP Ranking  : '+str(blup_avg)+"\n")
    text.insert(END,'Average GB Ranking    : '+str(gb_avg)+"\n")
    text.insert(END,'Average Lasso Ranking : '+str(la_avg)+"\n")
    text.insert(END,'Average RF Ranking    : '+str(rf_avg)+"\n")
    text.insert(END,'Average Ridge Ranking : '+str(ri_avg)+"\n")
    text.insert(END,'Average SVM Ranking   : '+str(svm_avg)+"\n")
    height = [blup_avg, gb_avg, la_avg, rf_avg, ri_avg, svm_avg]
    bars = ('Average BLUP','Average GB','Average Lasso','Average Random Forest','Average Ridge','Average SVM',)
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
    
def traitsGraph():
    r = {}
    for name in names:
        r[name] = round(np.mean([ranks[method][name] for method in ranks.keys()]), 2)
 
    methods = sorted(ranks.keys())
    ranks["Mean"] = r
    methods.append("Mean")

    meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])

    meanplot = meanplot.sort_values('Mean Ranking', ascending=False)
    sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar", size=14, aspect=1.9, palette='coolwarm')
    plt.show()
    
font = ('times', 14, 'bold')
title = Label(main, text='An Evaluation of Machine-Learning for Predicting Phenotype: Studies in Yeast, Rice, and Wheat',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Dataset", command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

modelButton = Button(main, text="Process Dataset Features", command=processDataset)
modelButton.place(x=50,y=200)
modelButton.config(font=font1)

psoBPButton = Button(main, text="Run All 5 Machine Learning Algorithms", command=runAlgorithms)
psoBPButton.place(x=50,y=250)
psoBPButton.config(font=font1)

predictButton = Button(main, text="Average Ranking Graph", command=rankingGraph)
predictButton.place(x=50,y=300)
predictButton.config(font=font1)

graphButton = Button(main, text="All Traits Graph", command=traitsGraph)
graphButton.place(x=50,y=350)
graphButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500,y=100)
text.config(font=font1)


main.config(bg='magenta3')
main.mainloop()
