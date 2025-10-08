import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import  confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Load the training dataset
train_data = pd.read_csv('sign_mnist_13bal_train.csv')

# Separate the data (features) and the  classes
X_train = train_data.drop('class', axis=1)  # Features (all columns except the first one)
X_train = X_train / 255.0 # pixels are scaled from 0-255 to 0-1
y_train = train_data['class']   # Target (first column)

# Load the testing dataset
test_data = pd.read_csv('sign_mnist_13bal_test.csv')

# Separate the data (features) and the  classes
# basically validate is test
X_validate = test_data.drop('class', axis=1)  # Features (all columns except the first one)
X_validate = X_validate / 255.0
y_validate = test_data['class']   # Target (first column)

# Use this line to get you started on adding a validation dataset
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=40, random_state=0)
# tol tells the neural network to stop training when the improvements in loss are smaller than 0.005 between iterations
neural_net_model = MLPClassifier( hidden_layer_sizes=(20),random_state=42,tol=0.005)

neural_net_model.fit(X_train, y_train)
# Determine model architecture 
# weight matrices
# shape = neurons in previous layer * neurons in current layer
layer_sizes = [neural_net_model.coefs_[0].shape[0]]  # Start with the input layer size
# creates list of input neurons, hidden neurons, and output neurons
layer_sizes += [coef.shape[1] for coef in neural_net_model.coefs_]  # Add sizes of subsequent layers
# presents it as 784 x 20 x 13
layer_size_str = " x ".join(map(str, layer_sizes))
# how many labels
print(f"Training set size: {len(y_train)}")
print(f"Layer sizes: {layer_size_str}")


# predict the classes from the training and test sets
# do on both to check if there is overfitting
y_pred_train = neural_net_model.predict(X_train) # tells how good it memorizes
y_pred = neural_net_model.predict(X_validate)

# Create dictionaries to hold total and correct counts for each class
correct_counts = defaultdict(int)
total_counts = defaultdict(int)
overall_correct = 0

# Count correct test predictions for each class
# this is going through the validation set
for true, pred in zip(y_validate, y_pred): # zip pairs true value with predicted value
    total_counts[true] += 1 # count of samples per this class
    if true == pred:
        correct_counts[true] += 1
        overall_correct += 1

# For comparison, count correct _training_ set predictions
total_counts_training = 0
correct_counts_training = 0
for true, pred in zip(y_train, y_pred_train): # pairs true variables from the actual training data with the model's predicted class for that sample
    total_counts_training += 1
    if true == pred:
        correct_counts_training += 1

# low training accuracy --> underfit and hasn't learned patterns


# Calculate and print accuracy for each class and overall test accuracy
for class_id in sorted(total_counts.keys()): # classes in the validation set in order
    accuracy = correct_counts[class_id] / total_counts[class_id] *100
    print(f"Accuracy for class {class_id}: {accuracy:3.0f}%") # accuracy for each class
print(f"----------")
overall_accuracy = overall_correct / len(y_validate)*100
print(f"Overall Validation Accuracy: {overall_accuracy:3.1f}%")
overall_training_accuracy = correct_counts_training / total_counts_training*100
print(f"Overall Training Accuracy: {overall_training_accuracy:3.1f}%")
print("The validation accuracy has increased by 47.7% and the training accuracy has increased by 79.3%.")
conf_matrix = confusion_matrix(y_validate, y_pred) # how many predictions were correct/misclassified
class_ids = sorted(total_counts.keys())

# For better formatting
print("Confusion Matrix:")
print(f"{'':9s}", end='') # 9 spaces at the start of the row so first column of row labels has space
for label in class_ids:
    print(f"Class {label:2d} ", end='') # prints class headers
print()  # Newline for next row

for i, row in enumerate(conf_matrix): # in each row of confusion matrix, 
    print(f"Class {class_ids[i]}:", " ".join(f"{num:8d}" for num in row))
# off diagnol counts
print("The model had the most difficulty identifying Class 1 ('B') and Class 8 ('I').")
