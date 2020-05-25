
#import dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston


#understanding the data
boston = load_boston()
data=pd.DataFrame(boston.data,columns=boston.feature_names)
data["price"]= boston.target
print(data.describe())


#ACCESS DATA ATTRIBUTES
dataset = boston.data
for name,index in enumerate(boston.feature_names):
    print(index,name)


#reshaping data
data=dataset[:,12].reshape(-1,1)


#shape of the data
np.shape(data)


#target values
target = boston.target.reshape(-1,1)


#shape of target
np.shape(target)


# ensuring that matplotlib is working inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data,target,color="green")
plt.xlabel('Lower income population')
plt.ylabel('cost of House')
plt.show()


from sklearn.linear_model import LinearRegression

#creating a regression model
reg = LinearRegression()

# fit the model
reg.fit(data,target)


# prediction
pred = reg.predict(data)



# ensuring that matplotlib is working inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data,target,color="green")
plt.plot(data,pred,color="red")
plt.xlabel('Lower income population')
plt.ylabel('cost of House')
plt.show()


# circumventing the curve issue by polynomial model
from sklearn.preprocessing import PolynomialFeatures

# to allow merging of models
from sklearn.pipeline import make_pipeline


model=make_pipeline(PolynomialFeatures(3),reg)

#fitting the model

model.fit(data,target)

pred=model.predict(data)


# ensuring that matplotlib is working inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data,target,color="green")
plt.plot(data,pred,color="red")
plt.xlabel('Lower income population')
plt.ylabel('cost of House')
plt.show()


