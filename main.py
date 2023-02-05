#import modules
from sklearn.linear_model import Perceptron as Perception 

#AND gate train data
X = [[0,0],[0,1],[1,0],[1,1]]

y = [0,0,0,1]
 
#Create a model
model = Perception()

#Train the model
model.fit(X,y)

#Test the model
print(model.predict([[0,0],[0,1],[1,0],[1,1]]))

#view the predicted data
print(model.predict([[0,0],[0,1],[1,0],[1,1]]))

#view the accuracy of the model
print(model.score(X,y))

