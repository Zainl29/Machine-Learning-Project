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


## Preprocessing: 
