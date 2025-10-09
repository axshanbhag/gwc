import pandas as pd
import GWCutilities as util
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_text
# all columns and content is shown
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

print("\n-----\n")

#Create a variable to read the dataset
df = pd.read_csv("heartDisease_2020_sampling.csv")

print(
    "We will be performing data analysis on this Indicators of Heart Disease Dataset. Here is a sample of it: \n"
)

#Print the dataset's first five rows
print(df.head())

input("\n Press Enter to continue.\n")



#Data Cleaning
#Label encode the dataset

le = LabelEncoder()

# List of columns you want to encode
cols_to_encode = ["HeartDisease", "GenHealth", "Smoking", "AlcoholDrinking", "PhysicalActivity", "AgeCategory"]

# Apply encoding to each column
# label encoding when order matters
for col in cols_to_encode:
    df[col] = le.fit_transform(df[col])

print("\nHere is a preview of the dataset after label encoding. \n")
print(df.head())

input("\nPress Enter to continue.\n")

#One hot encode the dataset
# for binary things
df = pd.get_dummies(df, columns=["Sex", "Race"], drop_first=False)
print(
    "\nHere is a preview of the dataset after one hot encoding. This will be the dataset used for data analysis: \n"
)
print(df.head())


input("\nPress Enter to continue.\n")



#Creates and trains Decision Tree Model
from sklearn.model_selection import train_test_split
# X is all of the columns excpet target
X = df.drop("HeartDisease", axis = 1)
y = df["HeartDisease"] # target column of 0/1
# data is split into training and desting sets
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 1)
# depth is limited to prevent overfitting, fits tree on the trained values
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth = 10, class_weight = 'balanced')
clf = clf.fit(X_train, y_train)





#Test the model with the testing data set and prints accuracy score
test_predictions = clf.predict(X_test)

from sklearn.metrics import accuracy_score
test_acc = accuracy_score(y_test, test_predictions)
print("The accuracy with the testing data set of the Decision Tree is :" + str(test_acc))


#Prints the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, test_predictions, labels = [1,0])
print(cm)




#Test the model with the training data set and prints accuracy score
train_predictions = clf.predict(X_train)

from sklearn.metrics import accuracy_score
# compares predicted vs actual labels
train_acc = accuracy_score(y_train, train_predictions)
print("The accuracy with the testing data set of the Decision Tree is :" + str(train_acc))



input("\nPress Enter to continue.\n")



#Prints another application of Decision Trees and considerations
print("\nBelow is another application of decision trees and considerations for using them:\n")
print("You can use decision trees when you need to make decisions that involve a number of factors such as the outfit you plan to wear for a school day. When creating the decisions, you need to consider all influential factors to gain a holisitic perspective. For example, when deciding your outfit, you need to consider the weather, the classroom temperature, events happening during school, dress code, etc.")




#Prints a text representation of the Decision Tree
print("\nBelow is a text representation of how the Decision Tree makes choices:\n")
input("\nPress Enter to continue.\n")
# prints out a rule-based tree
tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)
