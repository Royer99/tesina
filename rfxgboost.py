import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from pylab import rcParams
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix


def plot_correlation(data):
    rcParams['figure.figsize'] = 15, 25
    fig = plt.figure()
    sns.heatmap(data.corr(), annot=True, fmt=".1f")
    fig.savefig('correlation_coefficient.png')


data = pd.read_csv('../tesina_local/dataset/balanced_dataset_ddos.csv')
# plot correlation matrix
# plot_correlation(data)

# get the labels
labels = data['Subcategory'].unique()
# get category
categories = data['Category'].unique()
# remove useless columns
del data['StartTime']
del data['LastTime']
del data['Category']

print(data.head())
print("Labels: ", labels)
# get statistical data
# print(data.describe())
# histogram
hist = data['Subcategory'].hist()
# plt.show()

# split data into train and test
X = data.drop('Subcategory', axis=1)
y = data['Subcategory']
# using 20% of the data for testing and 80% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train model
model = XGBClassifier(use_label_encoder=False)
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Confusion Matrix")
print(confusion_matrix(y_test, predictions))
print("Classification Report")
print(classification_report(y_test, predictions, zero_division=0))
