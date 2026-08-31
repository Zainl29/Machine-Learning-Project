# Machine-Learning-Project
Dataset available at: https://archive.ics.uci.edu/dataset/222/bank+marketing

### prerequisites:
1. Python
2. Spyder IDE

**This project is composed of two main parts:**


## Preprocessing: 

Before implementing the selected technique, we perform preprocessing on the dataset to ensure accurate predictions and avoid any errors. The data is split into numerical and categorical groups and dealt with individually.
  
- The ‘duration’ variable is dropped from the dataset, as this value is only known after a phone call ends and at which point variable ‘y’, which is we want to predict, is already known.
  
- The range and scale of values for each column differs greatly from each other and there's the existence of negative values. To allow all features to contribute equally to our model, we can ensure a consistent scale across all features while preserving the original distributions by using MinMaxScaler()
  
- Since columns are in text format, and the machine learning model requires numeric inputs, LabelEncoder() is used to convert them into numbers. This is more suitable for decision trees and assigns each category for categorical variables, a unique integer value



## Implementation: 

The decision tree classifier was selected for this dataset and is suitable because:

-	It can handle data of different types of both numeric and categorical
-	It’s an effective model for a tabular (.csv) dataset
-	It is common for classification tasks, which is the goal for this data
-	The data is labelled, so a supervised learning model is required
-	The relationships between the variables may be complex and non-linear, and decision trees can handle this type of data and model this
-	The model is easy to interpret and explainable

**Implementation.py:**

- The data is split randomly, where 80% becomes the training set that is used to build the tree by learning patterns, and 20% becomes the test set that’s used to evaluate performance
  
- ‘random_state’ is introduced to control the randomness of the split in the data to ensure that it remains constant every time the code runs
  
- The decision tree model is defined to have a maximum depth of 10, to control how many splits take place from the root node to the leaf node, to prevent ‘overfitting’, leading to inaccurate predictions for the test set
  
- There is a class imbalance for variable ‘y’, which is handled through the ‘class_weight’ parameter 


## Results:

The performance of the model is evaluated using a classification report that gives key metrics such as precision, recall, F1-score and support. A confusion matrix is also plotted that breaks predictions into true positives/negatives and false positives/negatives.
