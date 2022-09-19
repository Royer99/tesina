import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# read data
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
# split data into train and test
X = data.drop('Subcategory', axis=1)
y = data['Subcategory']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = MLPClassifier(solver='adam', activation="relu", alpha=1e-5,
                    hidden_layer_sizes=(20+4,), learning_rate='adaptive', verbose=True, early_stopping=True)
clf.fit(X_train, y_train)
# predict
y_pred = clf.predict(X_test)
# evaluate
print(classification_report(y_test, y_pred))
# confusion matrix
print(confusion_matrix(y_test, y_pred))
