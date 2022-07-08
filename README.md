# Heart_failure_Predictive_Model
This is a personal model for predicting heart failure. The intention of this model was just for practice data engineering, feature selection, and models' evaluation. This is only a brief summary of the work, you can find all the functions in the `heart_risk.py` script.

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
Use the following lines if you want to do a complete scatterplot of all features in the dataset
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
