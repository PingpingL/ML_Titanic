#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from captum.attr import IntegratedGradients


train = pd.read_csv(r'E:\File\2022_fall_study\Machine Learning\Final Project\train.csv')
test = pd.read_csv(r'E:\File\2022_fall_study\Machine Learning\Final Project\test.csv')

#clear training data
train = train.drop(columns=['PassengerId','Name','Ticket','Cabin'])
train_y = train['Survived']
train_label = train_y#create label
train_x = train.drop(['Survived'],axis=1)
train_x['Age'] = train_x['Age'].fillna(train_x["Age"].mean())#drop nan
train_x["Fare"] = train_x["Fare"].fillna(train_x["Fare"].mean())
b_sex = pd.get_dummies(train_x['Sex'])#change string to binary
combine_train_x = pd.concat((b_sex,train_x), axis=1)
combine_train_x = combine_train_x.drop(['Sex','Embarked','male'], axis=1)
trainx = combine_train_x.rename(columns={'female': 'Sex'})#1 is female,0 is male
train_feature = trainx
class_name = list(trainx.columns)
print(train_feature)
print(train_label)

#claer testing data
test = test.drop(columns=['PassengerId','Name','Ticket','Cabin','Embarked'])
b_sex = pd.get_dummies(test['Sex'])
combine_test = pd.concat((b_sex, test), axis=1)
combine_test = combine_test.drop(['male','Sex'], axis=1)
test_x = combine_test.rename(columns={"female": "Sex"})#1 is female,0 is male
print(test_x)


# In[205]:


#parameters
learning_rate = 0.001
num_epochs = 400
torch.manual_seed(1)
#model
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(6, 5)
        self.sigmoid1 = nn.ReLU()
        self.layer2 = nn.Linear(5, 4)
        self.sigmoid2 = nn.ReLU()
        self.layer3 = nn.Linear(4, 3)
        self.sigmoid3 = nn.ReLU()
        self.layer4 = nn.Linear(3, 2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out_linear1 = self.layer1(x)
        out_sigmoid1 = self.sigmoid1(out_linear1)
        out_linear2 = self.layer2(out_sigmoid1)
        out_sigmoid2 = self.sigmoid2(out_linear2)
        out_linear3 = self.layer3(out_sigmoid2)
        out_sigmoid3 = self.sigmoid3(out_linear3)
        out_linear4 = self.layer4(out_sigmoid3)
        out = self.softmax(out_linear4)
        return out
model = Model()

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#learning rate = 0.001

#traning loop
in_tensor = torch.from_numpy(np.asarray(train_feature)).type(torch.FloatTensor)
label_tensor = torch.from_numpy(np.asarray(train_label))
for epoch in range(num_epochs):
    #forward
    outputs = model(in_tensor)
    loss = criterion(outputs, label_tensor)
    #backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print
    if epoch % 50 == 0:
        print (f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.2f}')


# In[206]:


#Accuracy
prob_train = model(in_tensor).detach().numpy()
decision = np.argmax(prob_train, axis=1)
print("Accuracy:", sum(decision == train_label) / len(train_label))


# In[207]:


#Predict
test_in_tensor = torch.from_numpy(np.asarray(testx)).type(torch.FloatTensor)
prob_test = model(test_in_tensor).detach().numpy()
predict_test = np.argmax(prob_test, axis=1)
print("Predict survival rate is",sum(predict_test==1)/len(predict_test))


# In[208]:


#IntegratedGradients
fea_attr_ig = IntegratedGradients(model)
test_in_tensor.requires_grad_()
attr, delta = fea_attr_ig.attribute(test_in_tensor,target=1, return_convergence_delta=True)
attr = attr.detach().numpy()
attribution = pd.DataFrame(attr)
attribution = attribution.fillna(0)
average_attr = attribution.mean(axis=0)
print(average_attr)


# In[209]:


#Visualization
plt.figure(figsize=(6,6))
plt.bar([0,1,2,3,4,5], average_attr, align='center')
plt.xticks([0,1,2,3,4,5], class_name, wrap=True)
plt.ylabel('Attributes')
plt.title('Feature Attribute')

