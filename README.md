# Heart_failure_Predictive_Model
This is a personal model for predicting heart failure. The intention of this model was just for practice data engineering, feature selection, and models' evaluation. This is only a brief summary of the work, you can find all the functions in the `heart_risk.py` script. The dataset was taken from Kaggle: https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data

## Importing libraries
```
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
```

## Importing dataset
In this step, I used pandas to open the  `csv` file with all data
```
data = pd.read_csv(r'heart_failure_clinical_records_dataset.csv')
```
Now, you can create new variables or use the normal variables of the dataset. I create some interactions to see if these new features could improve the models performance.

```
new_col =  data['creatinine_phosphokinase'] * data['serum_creatinine']
data.insert(loc= 12, column = 'creatinine_interacts', value = new_col)

new_col =  data['age'] * data['serum_creatinine']
data.insert(loc= 13, column = 'agecreat_interacts', value = new_col)

new_col =  data['serum_sodium'] * data['ejection_fraction']
data.insert(loc= 14, column = 'sodiumeject_interacts', value = new_col)
```
## Data visualization
Use the following lines if you want to do a scatterplot of a specific pair of features of the dataset
```
fig, ax = plt.subplots()
colors = {0:'red', 1:'blue'}
ax.scatter(dataset['ejection_fraction'], dataset['time'], c=dataset['label'].map(colors))
plt.show()
```
Also, I wanted to use a correlation matrix to select the more appropriate features

```
corr_matrix= data.corr()
variable_names = list(data.columns)

plt.figure(figsize = (13,13),dpi = 300)
sn.heatmap(corr_matrix,cmap = 'inferno', annot=True)
plt.title('Heart Failure Correlation',
          fontsize= 16)
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()
```
![correlation_matrix](https://user-images.githubusercontent.com/87657676/178085891-fa4e02b4-f778-4e1c-87db-487ceffcc960.jpg)

Now, select the features that you want and split the training-testing set
```
features = select_corr_data(corr_matrix, -1, threshold = 0.20)
features.remove('DEATH_EVENT')

X = data.loc[:,features]
y = data.iloc[:,-1]
dataset = data.loc[:, features]
dataset['label'] = y

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
```
Please note that you can use different threshold for filtering features, different test_size, and random_state.

## Importing models from Sklearn and evaluate
### Logistic regression model
```
logit = LogisticRegression(solver = 'lbfgs', penalty = 'l2')

logit = logit.fit(X_train, y_train)
logit_y_pred = logit.predict(X_test)
```
### SVM classfier
I used gridsearch for the best estimator hyperparameters
```
param_grid = {'C': [1,2,3,4,5,6,7,8,9,10,15,20],
              'gamma': [0.001, 0.01, 0.1,0.51,1.1, 1.2, 1.3, 1.4, 1.5],
              'kernel': ['rbf', 'linear', 'poly']}

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)
print(grid.best_estimator_)

clf = SVC(C = 2, gamma = 0.1, kernel = 'rbf', probability = True)


clf = clf.fit(X_train, y_train)
svc_y_pred = clf.predict(X_test)


svc_metrics = create_metrics(y_test, svc_y_pred)
```
### KNN classifier
In this model, i also used gridsearch to obtain the better hyperparameters
```
param_grid = {'n_neighbors': [3,5,7,9,11,13,15],
              'weights': ['uniform', 'distance'],
              'algorithm': ['kd_tree', 'brute'],
              'p': [1,2]}

grid = GridSearchCV(KNeighborsClassifier(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)
print(grid.best_estimator_)

knn_clf = KNeighborsClassifier(n_neighbors= 5, weights = 'uniform'
                               ,algorithm = 'kd_tree', p = 1)

knn_clf =knn_clf.fit(X_train, y_train)

knn_y_pred = knn_clf.predict(X_test)

metrics = create_metrics(y_test, knn_y_pred)
```
Finally, I evaluate the models using accuracy, precision, and recall as metrics. It is widely known that false negatives in medical sciences are unacceptable, therefore, the best predictive model for classification was the KNN, because it is sacrificed some precision in comparison to other models, but the accuracy is higher and the recall is also 10 % higher in comparison to SVM and Logit. It is important to mention that the new variables created improved the recall of the KNN model, you can run the programm with no data transformations (interactions and ratios), and you should obtain ~0.75 of recall in this model.

![Engineered models](https://user-images.githubusercontent.com/87657676/178086240-f0b38808-e739-4315-8685-164cc4e15add.jpg)

