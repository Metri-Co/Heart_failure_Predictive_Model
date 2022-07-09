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
from sklearn.naive_bayes import GaussianNB

def remove_ids(array, var_position):
    ids = array[:, var_position]
    array = np.delete(array, var_position, axis=1)
    vector = np.ravel(array, "F")
    return vector, ids


def accuracy(groundtruth, predictions):
    tn, fp, fn, tp = confusion_matrix(groundtruth, predictions, labels=[0, 1]).ravel()
    obs = len(groundtruth)
    result = (tp + tn) / obs
    return result


def precision(groundtruth, predictions):
    # true positives / true positives + false positives
    tn, fp, fn, tp = confusion_matrix(groundtruth, predictions).ravel()
    result = tp / (tp + fp)
    return result


def recall(groundtruth, predictions):
    tn, fp, fn, tp = confusion_matrix(groundtruth, predictions).ravel()
    result = tp / (tp + fn)
    return result


def F1(groundtruth, predictions):
    numerator = precision(groundtruth, predictions) * recall(groundtruth, predictions)
    denominator = precision(groundtruth, predictions) + recall(groundtruth, predictions)
    result = 2 * (numerator / denominator)
    return result


def create_metrics(groundtruth, predictions):
    dic = {"accuracy": accuracy(groundtruth, predictions),
           "precision": precision(groundtruth, predictions),
           "recall": recall(groundtruth, predictions),
           "Fvalue": F1(groundtruth, predictions)}
    return dic
def normalize(df):
    import pandas as pd
    norm_df = df.copy()
    _, y = norm_df.shape

    for i in range(y):
        min_i = min(norm_df.iloc[:,i])
        max_i = max(norm_df.iloc[:,i])

        norm_df = norm_df.apply(lambda x: (x - min_i)/(max_i - min_i), axis = 1)

    return norm_df


def select_corr_data(corr_mat, y_index, threshold = 0.4):
    """
    Function for selecting the more correlated data given a corr_matrix
    :param corr_mat: Correlation matrix (dataframe)
    :param y_index: index of the desired feature to predict
    :param threshold: threshold of correlation index value
    :return: list of selected data
    """
    x, y = corr_mat.shape
    features = []
    for j in range(y):
        i = y_index
        corr_index = corr_mat.iloc[i,j]
        if np.absolute(corr_index) >= threshold:
            features.append(corr_mat.columns.tolist()[j])

    return features


def roc_curve(groundtruth, probabilities, predictions, estimator_name=str):
    """
    Function for plotting the ROC curve and calculating the AUC

    Parameters
    ----------
    groundtruth : List or 1D array
        The real values of the dataset.
    predictions : List or 1D array
        The predicted values by the classifier.
    estimator_name : string
        Name of the classifier, it will be printed in the Figure

    Returns
    -------
    Figure.
    AUC.
    """
    from sklearn.metrics import roc_auc_score
    sensitivities = []
    especificities = []

    sensitivities.append(1)
    especificities.append(1)

    thresholds = [i * 0.05 for i in range(1, 10, 1)]
    for t in thresholds:
        prob = probabilities[:, 1]
        prob = np.where(prob >= t, 1, 0)
        recall_data = recall(groundtruth, prob)
        precision_data = precision(groundtruth, prob)
        sensitivities.append(recall_data)
        espc = 1 - precision_data
        especificities.append(espc)
    sensitivities.append(0)
    especificities.append(0)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(especificities, sensitivities, marker='o', linestyle='--', color='r')
    plt.plot([i * 0.01 for i in range(100)], [i * 0.01 for i in range(100)])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title(f'{estimator_name} ROC curve')
    plt.savefig(f'{estimator_name} ROC curve.jpg', dpi=300)

    AUC = roc_auc_score(groundtruth, predictions)
    return AUC


#%%
data = pd.read_csv(r'heart_failure_clinical_records_dataset.csv')

new_col =  data['creatinine_phosphokinase'] * data['serum_creatinine']
data.insert(loc= 12, column = 'creatinine_interacts', value = new_col)

new_col =  data['age'] * data['serum_creatinine']
data.insert(loc= 13, column = 'agecreat_interacts', value = new_col)

new_col =  data['serum_sodium'] * data['ejection_fraction']
data.insert(loc= 14, column = 'sodiumeject_interacts', value = new_col)

corr_matrix= data.corr()
variable_names = list(data.columns)

plt.figure(figsize = (13,13),dpi = 300)
sn.heatmap(corr_matrix,cmap = 'inferno', annot=True)
plt.title('Heart Failure Correlation',
          fontsize= 16)
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()
plt.savefig('correlation_matrix.jpg', dpi=300)

features = select_corr_data(corr_matrix, -1, threshold = 0.20)
features.remove('DEATH_EVENT')

X = data.loc[:,features]
y = data.iloc[:,-1]
dataset = data.loc[:, features]
dataset['label'] = y

# visualizing data according to the class
#fig, ax = plt.subplots()
#colors = {0:'red', 1:'blue'}
#ax.scatter(dataset['ejection_fraction'], dataset['time'], c=dataset['label'].map(colors))
#plt.show()


scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
#%% Logistic regression model
logit = LogisticRegression(solver = 'lbfgs', penalty = 'l2')
log_scores = cross_val_score(logit, X, y,cv = 10)

logit = logit.fit(X_train, y_train)
logit_proba = logit.predict_proba(X_test)
logit_y_pred = logit.predict(X_test)

logit_metrics = create_metrics(y_test, logit_y_pred)
logit_auc = roc_curve(y_test, logit_proba, logit_y_pred, estimator_name= 'Logit class engin')
#%% SVM classfier
# Grid search
param_grid = {'C': [1,2,3,4,5,6,7,8,9,10,15,20],
              'gamma': [0.001, 0.01, 0.1,0.51,1.1, 1.2, 1.3, 1.4, 1.5],
              'kernel': ['rbf', 'linear', 'poly']}

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)
print(grid.best_estimator_)

clf = SVC(C = 2, gamma = 0.1, kernel = 'rbf', probability = True)
svc_scores = cross_val_score(clf, X, y, cv = 10)

clf = clf.fit(X_train, y_train)
svc_proba = clf.predict_proba(X_test)
svc_y_pred = clf.predict(X_test)


svc_metrics = create_metrics(y_test, svc_y_pred)
svc_auc = roc_curve(y_test, svc_proba, svc_y_pred, estimator_name= 'SVM eng')

#%% KNN classifier

param_grid = {'n_neighbors': [3,5,7,9,11,13,15],
              'weights': ['uniform', 'distance'],
              'algorithm': ['kd_tree', 'brute'],
              'p': [1,2]}

grid = GridSearchCV(KNeighborsClassifier(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)
print(grid.best_estimator_)

knn_clf = KNeighborsClassifier(n_neighbors= 5, weights = 'uniform'
                               ,algorithm = 'kd_tree', p = 1)
knn_scores = cross_val_score(knn_clf, X, y, cv = 10)

knn_clf =knn_clf.fit(X_train, y_train)

knn_y_pred = knn_clf.predict(X_test)
knn_proba = knn_clf.predict_proba(X_test)

metrics = create_metrics(y_test, knn_y_pred)
knn_auc = roc_curve(y_test, knn_proba, knn_y_pred, estimator_name= 'KNN class non-engin')

#%%
x_axis = np.arange(0,3)

accuracies = [logit_metrics['accuracy'], svc_metrics['accuracy'], metrics['accuracy']]
precisions = [logit_metrics['precision'], svc_metrics['precision'], metrics['precision']]
recalls = [logit_metrics['recall'], svc_metrics['recall'], metrics['recall']]


plt.figure(figsize = (8,5), dpi = 300)
plt.bar(x=x_axis - 0.25, height = accuracies, width = 0.25, label ='accuracy', color = 'red')
plt.bar(x=x_axis, height = precisions, width = 0.25, label ='precision', color = 'blue')
plt.bar(x=x_axis + 0.25, height = recalls, width = 0.25, label ='recall', color = 'orange')

plt.ylabel('Performance (%)')
plt.legend(loc = 4)
plt.xticks(x_axis, ['Logit Class', 'SVM', 'KNN'])
plt.title("Models' performance with no data transformation")
plt.ylim([0,1.1])
plt.savefig('Non engineered models.jpg', dpi =300)