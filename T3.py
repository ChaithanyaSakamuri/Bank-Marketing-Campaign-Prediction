import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
# Load the dataset - make sure the path is correct and uses raw string or double backslashes
data = pd.read_csv("bank.csv", sep=';')
# Encode categorical variables
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])
# Split features and target
X = data.drop('y', axis=1)
y = data['y']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create and train the decision tree model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate model performance
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["Actual No", "Actual Yes"], columns=["Predicted No", "Predicted Yes"])
print("\nConfusion Matrix:")
print(cm_df)
# Classification report
cr = classification_report(y_test, y_pred, output_dict=True)
cr_df = pd.DataFrame(cr).transpose()
print("\nClassification Report:")
print(cr_df)
# Visualize the decision tree
plt.figure(figsize=(20, 8))
plot_tree(model, 
          feature_names=X.columns, 
          class_names=["No", "Yes"], 
          filled=True, 
          rounded=True,
          proportion=True,
          fontsize=10)
plt.title("Decision Tree - Bank Marketing Dataset")
plt.show()
