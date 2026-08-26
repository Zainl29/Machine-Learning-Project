# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


df = pd.read_csv("C:\\Users\\zainl\\Downloads\\Preprocessing-bank-additional-full.csv")

# Define the features (X) and the target variable (y)
X = df.drop('y', axis=1)
y = df['y']

# Split the dataset into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


tree_clf = DecisionTreeClassifier(max_depth = 10, random_state = 42, class_weight="balanced")
tree_clf.fit(X_train, y_train)


y_pred = tree_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()

report = classification_report(y_test, y_pred)
print(report)






