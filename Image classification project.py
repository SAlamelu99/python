#import dependencies for image classifier

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets
digits=datasets.load_digits()
data=pd.DataFrame(digits.data)


#viewing column heads
print(data.head())





#extractind data from the dataset and viewing them up close
a=data.iloc[2,:].values
print(a)
a.shape



#reshaping the extracted data into a reasonable size
a=a.reshape(8,8).astype('uint8')
a.shape
print(a)
plt.imshow(a,cmap="Blues_r")


#preparing the data
#preparing labels and data values
df_x=data.iloc[:,2:]
df_y=data.iloc[:,1]


#creating test and train sizes/batches
x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=4)


#check data
y_train.head()



#calling rf classifier
rf = RandomForestClassifier(n_estimators=100)



#fit the model
rf.fit(x_train, y_train)



#predict on test data
pred = rf.predict(x_test)


pred


#check prediction accuracy 

s = y_test.values
print(s)

#calculate number of correctly predicted values
count=0
for i in range(len(pred)):
    if pred[i]==s[i]:
        count=count+1;


count


#total values that the prediction code was run on
len(pred)


#accuracy value
count/len(pred)



