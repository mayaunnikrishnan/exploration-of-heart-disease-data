# HeartLens: Predicting Heart Disease with Random Forest Insights
## Overview
Heart disease is a leading cause of death in the United States, affecting people of various races. The dataset used in this project consists of annual survey data collected by the CDC, encompassing the health status of over 400,000 adults. Key indicators of heart disease include high blood pressure, high cholesterol, smoking, diabetes status, obesity (high BMI), insufficient physical activity, and excessive alcohol consumption. Identifying and addressing these risk factors are crucial in healthcare, and machine learning methods can aid in detecting patterns to predict a patient's condition.
![Exploring Heart Disease Data](https://github.com/mayaunnikrishnan/exploration-of-heart-disease-data/assets/128244858/41bab921-31c1-48a9-b251-1eb1facb61a4)

## Table of Contents
- [Reading Data Set](#reading-data-set)
- [Summary of Features](#summary-of-features)
- [Data Cleaning](#data-cleaning)
- [Exploring Data Analysis and Visualization](#exploring-data-analysis-and-visualization)
- [Data Preprocessing](#data-preprocessing)
  - [Label Encoding](#label-encoding)
  - [Train Test Split](#train-test-split)
  - [Standardising the Data Set](#standardising-the-data-set)
- [Modeling](#modeling)
  - [Finding the Best Model Using Grid Search CV](#finding-the-best-model-using-grid-search-cv)
  - [Model of Random Forest Classifier](#model-of-random-forest-classifier)
  - [Prediction of the Model](#prediction-of-the-model)
  - [Export the Test Model to a Pickle File](#export-the-test-model-to-a-pickle-file)
- [Streamlit Machine Learning Model App](#streamlit-machine-learning-model-app)

     -Visit the Heart Disease Data Exploration Web App [here](https://exploration-of-heart-disease-data-76wdyy6nse6v9i3at78ixt.streamlit.app/).

- [Conclusion](#conclusion)

  ## Reading Data Set
The dataset used in this project is sourced from [kamilpytlak/personal-key-indicators-of-heart-disease](https://www.kaggle.com/kamilpytlak/personal-key-indicators-of-heart-disease).

## Summary of Features

- **HeartDisease**: Binary variable indicating the presence or absence of heart disease.
- **BMI**: Body Mass Index, a measure of body fat based on height and weight.
- **Smoking**: Smoking status of the respondent.
- **AlcoholDrinking**: Alcohol consumption habits of the respondent.
- **Stroke**: History of stroke.
- **PhysicalHealth**: Self-reported physical health status.
- **MentalHealth**: Self-reported mental health status.
- **DiffWalking**: Difficulty in walking.
- **Sex**: Gender of the respondent.
- **AgeCategory**: Age category of the respondent.
- **Race**: Race or ethnicity of the respondent.
- **Diabetic**: Diabetes status of the respondent.
- **PhysicalActivity**: Level of physical activity.
- **GenHealth**: General health status.
- **SleepTime**: Duration of sleep.
- **Asthma**: Asthma status of the respondent.
- **KidneyDisease**: Kidney disease status of the respondent.
- **SkinCancer**: Skin cancer status of the respondent.
## Data Cleaning

The initial dataset may suffer from class imbalance, particularly in the target variable indicating the presence or absence of heart disease. This can lead to biased model predictions. To address this issue, balancing techniques can be applied.

### Class Imbalance
- **Initial State**: The dataset may exhibit an imbalance between the classes of the target variable, with one class significantly outnumbering the other.
- **Impact**: Imbalance can lead to biased model predictions, where the minority class is often overlooked or misclassified.
- **Solution**: Various techniques can be employed to balance the dataset, such as:
  - **Random Oversampling**: Increasing the number of instances in the minority class by duplicating samples.
  - **Random Undersampling**: Decreasing the number of instances in the majority class by removing samples.
  - **SMOTE (Synthetic Minority Over-sampling Technique)**: Generating synthetic samples for the minority class to achieve a balanced dataset.
  - **PD.Concat Method**: Concatenating instances from the minority and majority classes to create a balanced dataset.

### Balancing with `pd.concat` Method
- **Approach**: By concatenating instances from the minority and majority classes, the `pd.concat` method creates a new balanced dataset.
- **Implementation**: 
  ```python
  # Assuming df_minority and df_majority are dataframes containing instances of minority and majority classes respectively
  balanced_df = pd.concat([df_minority, df_majority])
## Exploratory Data Analysis and Visualization

Exploratory data analysis (EDA) is conducted to gain insights into the relationships between variables in the dataset through visualizations and statistical summaries.

### Correlation Graph Observations

A correlation graph provides insights into the relationships between variables. Key observations include:
- **Physical Activity**: Shows an inverse relationship with other variables in the dataset.
- **Difficulty Walking and Stroke**: Exhibit a more direct proportional relationship with heart disease.
- **Physical Health and Diabetes**: Have stronger correlations with heart disease, followed by skin cancer.
- **Mental Health**: Shows a stronger correlation with difficulty walking than with physical health.
- **Smoking**: Is more strongly correlated with physical health and difficulty walking than with diabetes.
- **Alcohol Drinking**: Exhibits a stronger correlation with smoking than with asthma.
- **Kidney Disease**: Correlates with physical health, difficulty walking, and diabetes.

### Detailed Visual Analysis

Visualizations are created to explore relationships between variables in more detail. Each category is analyzed individually to understand its impact on heart disease and its correlation with other variables.

For detailed visualizations and insights, refer to the exploratory data analysis section in the project repository.
## Data Preprocessing

Data preprocessing is essential to prepare the dataset for modeling, involving steps such as handling categorical variables, splitting the data into training and testing sets, and standardizing the features.

### Label Encoding

Categorical columns are converted into numerical format using label encoding to facilitate modeling. This ensures that categorical variables can be used as input for machine learning algorithms.

### Train Test Split

The dataset is divided into training and testing sets using the train_test_split function from the sklearn.model_selection module. This helps evaluate the model's performance on unseen data.

```python
from sklearn.model_selection import train_test_split

# Assuming X contains features and y contains the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Modeling

### Finding the Best Model Using Grid Search CV
Grid Search Cross-Validation (GridSearchCV) is employed to determine the optimal hyperparameters for our models. The process involves exhaustively searching through a specified parameter grid and selecting the combination of parameters that yield the highest cross-validation score.

### Model of Random Forest Classifier
A Random Forest Classifier model is implemented for predicting heart disease. Random forests are ensemble learning methods that construct a multitude of decision trees during training and output the mode of the classes for classification tasks. This model is chosen based on its performance during hyperparameter tuning using GridSearchCV.

### Prediction of the Model
The trained Random Forest Classifier model is used to make predictions on the test set. By applying the model to unseen data, we evaluate its performance and assess its ability to generalize to new instances.

### Export the Trained Model to a Pickle File
After training the Random Forest Classifier model, it is serialized and saved to a `.pkl` file using the pickle library. This allows for easy storage and later retrieval of the trained model for deployment or further analysis.

## Streamlit Machine Learning Model App
A Streamlit web application is developed to provide a user-friendly interface for predicting heart disease based on user input. The app utilizes the trained Random Forest Classifier model to make predictions in real-time. Users can input relevant features such as BMI, smoking status, physical activity level, and other indicators, and receive a prediction regarding the likelihood of heart disease. The app serves as a practical tool for individuals to assess their risk of heart disease and take preventive measures accordingly.

## Conclusion
In this project, we utilized machine learning techniques to predict heart disease based on various risk factors. By employing GridSearchCV for hyperparameter tuning and selecting the Random Forest Classifier model, we achieved promising results in terms of predictive accuracy. The developed Streamlit web application offers a convenient means for individuals to assess their risk of heart disease and make informed health decisions. Further enhancements and refinements to the model and application can be explored in future iterations of the project.
