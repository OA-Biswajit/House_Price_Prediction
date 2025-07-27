House Price Prediction: Project Documentation

Project: A machine learning model to predict house sale prices based on property features.

1. Project Overview
The goal of this project is to build a regression model that accurately predicts the Sale Price of a house. The model is trained on a dataset containing various features of previously sold houses, such as their area, year built, and overall condition. By learning the patterns in this historical data, the model can estimate the price of new, unseen houses.

Dataset: HousePricePrediction.xlsx

Model Used: Random Forest Regressor

Primary Libraries: pandas, scikit-learn, matplotlib, seaborn

2. Machine Learning Workflow
The project follows a standard machine learning pipeline, broken down into five main stages:

Data Loading and Exploration: Importing the data and performing an initial analysis to understand its structure and contents.

Data Visualization: Creating plots like a correlation heatmap to visually explore relationships between different features.

Data Preprocessing: Cleaning the data by handling missing values and converting categorical (text-based) features into a numerical format suitable for the model.

Model Building and Training: Splitting the data into training and testing sets, initializing a Random Forest Regressor, and training it on the training data.

Model Evaluation & Prediction: Using the trained model to make predictions on the unseen test data, evaluating its performance, and finally, using it to predict the price of new sample data.

3. Detailed Code Walkthrough
This section provides a line-by-line explanation of the Python code used in the predict_prices.ipynb notebook.

Part 1: Setup and Data Loading
1.1 - Importing Necessary Libraries
This first block of code imports all the essential Python libraries needed for the project.

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import pandas as pd: Imports the pandas library for data manipulation and analysis. pd is the standard alias.

import numpy as np: Imports NumPy for numerical operations.

import seaborn as sns & import matplotlib.pyplot as plt: Imports Python's primary data visualization libraries.

from sklearn.model_selection import train_test_split: Imports the function to split our dataset into training and testing subsets.

from sklearn.ensemble import RandomForestRegressor: Imports the Random Forest algorithm for regression tasks.

from sklearn.metrics import ...: Imports functions to evaluate the model's performance.

1.2 - Loading the Dataset
Here, we load the data from the Excel file into a pandas DataFrame.

# --- Step 1: Load and Explore the Data ---
house = pd.read_excel("HousePricePrediction.xlsx")
house.head()

house = pd.read_excel(...): Reads the specified Excel file and loads its contents into a DataFrame named house.

house.head(): Displays the first 5 rows of the DataFrame for a quick preview.

Part 2: Data Preprocessing & Visualization
2.1 - Handling Missing Values
We fill missing numerical values with the median of their respective columns.

# Handle Missing Values (if any)
for col in house.select_dtypes(include=np.number).columns:
    if house[col].isnull().sum() > 0:
        median_val = house[col].median()
        house[col].fillna(median_val, inplace=True)

for col in ...: This loop iterates through only the numerical columns.

if house[col].isnull().sum() > 0:: Checks if the column has any missing values.

median_val = ...: Calculates the median value for that column.

house[col].fillna(...): Replaces all null values in the column with the median.

2.2 - Converting Categorical Features (One-Hot Encoding)
This step converts text-based columns into a numerical format.

# Handle Categorical Features using One-Hot Encoding
categorical_cols = house.select_dtypes(include='object').columns
df_model = pd.get_dummies(house, columns=categorical_cols, drop_first=True)

categorical_cols = ...: Identifies all text-based columns.

df_model = pd.get_dummies(...): Performs one-hot encoding, creating new binary (0 or 1) columns for each category. drop_first=True is used to avoid data redundancy.

2.3 - Visualizing Feature Correlations with a Heatmap
A heatmap helps us understand the relationships between different numerical features. It shows which variables tend to move together.

# Visualize correlations
plt.figure(figsize=(12, 10))
sns.heatmap(df_model.corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap of Features')
plt.show()

plt.figure(figsize=(12, 10)): Creates a new figure (a canvas) for our plot, making it 12 inches wide by 10 inches tall for better readability.

df_model.corr(): This is the core of the heatmap. The .corr() method calculates the correlation coefficient (a value between -1 and 1) for every pair of numerical columns in the df_model DataFrame.

A value near +1 (bright red on our map) means a strong positive correlation (as one feature increases, the other tends to increase).

A value near -1 (bright blue) means a strong negative correlation (as one increases, the other tends to decrease).

A value near 0 (light color) means little to no correlation.

sns.heatmap(...): This Seaborn function takes the correlation matrix and visualizes it as a color-encoded map.

cmap='coolwarm': Sets the color scheme. 'coolwarm' is a good choice as it uses distinct colors (blue and red) for negative and positive correlations.

annot=False: This is set to False to not write the actual correlation values on the map, which would make it too crowded. If you set it to True, you would see the numbers inside each square.

Part 3: Model Training and Evaluation
3.1 - Separating and Splitting the Data
We define our features (x) and target (y) and then split them into training and testing sets.

# Prepare data for modeling
x = df_model.drop('Sale Price', axis=1)
y = df_model['Sale Price']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x = ... and y = ...: Separates the data into features (inputs) and the target (output).

train_test_split(...): Splits the data, holding back 20% (test_size=0.2) for final evaluation. random_state=42 ensures the split is the same every time.

3.2 - Building and Training the Model
We initialize and train the Random Forest Regressor.

# --- Step 3: Model Building and Training ---
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(x_train, y_train)

model = RandomForestRegressor(...): Initializes the model.

model.fit(x_train, y_train): Trains the model by showing it the training features and the corresponding correct answers (sale prices).

3.3 - Evaluating the Model
We check how well our model performs on the unseen test data.

# --- Step 4: Model Evaluation ---
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

y_pred = model.predict(x_test): Makes predictions on the test set.

mean_absolute_error(...) and r2_score(...): Calculate performance metrics by comparing predictions (y_pred) to the actual prices (y_test).

Part 4: Feature Importance and Sample Predictions
4.1 - Visualizing Feature Importance
This plot shows which features the model found most influential.

# --- Feature Importance ---
importances = model.feature_importances_
feature_importance_df = pd.DataFrame(...)
sns.barplot(data=feature_importance_df.head(15), ...)
plt.show()

model.feature_importances_: Extracts the importance score for each feature from the trained model.

The code then creates a DataFrame and plots a bar chart of the top 15 features, giving us insight into what drives the model's predictions.

4.2 - Predicting Prices for New Sample Data
This final section demonstrates how to use the trained model to predict the price of new, unseen data. This is the most critical step for real-world application.

# Select two samples from the original data
sample_tests = house.sample(2, random_state=42)

sample_tests = house.sample(2, ...): We select two random rows from the original house DataFrame to simulate new data. random_state=42 ensures we get the same two samples every time.

# Apply the same preprocessing steps as the training data
# ... (code for handling missing values) ...
categorical_cols = sample_tests.select_dtypes(include='object').columns
sample_tests_processed = pd.get_dummies(sample_tests, columns=categorical_cols, drop_first=True)

Crucially, we apply the exact same preprocessing steps (filling missing values, one-hot encoding) to our new sample data as we did to our training data. The model was trained on data in a specific format and will only work if new data is in that same format.

# Ensure the sample test data has the same columns as the training data
missing_cols = set(x_train.columns) - set(sample_tests_processed.columns)
for c in missing_cols:
    sample_tests_processed[c] = 0

sample_tests_processed = sample_tests_processed[x_train.columns]

This is a vital alignment step. Our small sample of two houses might not have all the possible categories that were in the full training set (e.g., maybe neither sample has Configuration_FR3). After one-hot encoding, our sample data would be missing that column.

missing_cols = set(x_train.columns) - ...: This line finds any columns that are in our training data (x_train) but are not in our newly processed sample data.

for c in missing_cols: ...: The loop adds these missing columns to our sample data and fills them with 0. This tells the model that these categories are not present in the new samples.

sample_tests_processed = sample_tests_processed[x_train.columns]: This final line ensures that the columns in our sample data are in the exact same order as the columns in the training data. The model is sensitive to column order.

# Predict the sale price
predicted_prices = model.predict(sample_tests_processed)

predicted_prices = model.predict(...): Now that our sample data is perfectly formatted, we can feed it to the model to get our price predictions.

# Display the original sample data and the predicted prices
print("Sample Test Cases:")
display(sample_tests)
print("\nPredicted Sale Prices:")
for i, price in enumerate(predicted_prices):
    print(f"Sample {i+1}: ${price:,.2f}")

Finally, we display the original sample houses and their predicted prices, formatted nicely for readability.