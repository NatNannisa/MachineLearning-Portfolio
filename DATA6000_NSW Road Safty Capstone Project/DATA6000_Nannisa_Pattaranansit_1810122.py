
## Descriptive Part ##

import requests
import pandas as pd
import numpy as np

# URL of the Excel file
url = "https://www.bitre.gov.au/sites/default/files/documents/bitre_fatalities_nov2024.xlsx"

# Fetch the file
response = requests.get(url)
with open("data.xlsx", "wb") as file:
    file.write(response.content)

# Load the specific sheet "BITRE_Fatality" and skip unnecessary rows
df = pd.read_excel("data.xlsx", sheet_name="BITRE_Fatality", skiprows=4)

# Filter rows where the 'State' column contains 'NSW'
df_nsw = df[df['State'] == 'NSW'].copy()

# =========================================
# Step 2: Handle Missing Values
# =========================================

# Replace specific missing values with NaN for easier handling
df_nsw.replace(-9, np.nan, inplace=True)
# Define a function to handle missing values based on column type
def handle_missing_values(df):
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            # Replace missing values in numerical columns with 0
            df[column].fillna(0, inplace=True)
        else:
            # Replace missing values in categorical columns with 'Unknown'
            df[column].fillna('Unknown', inplace=True)
    return df

# Apply the function to handle missing values
df_nsw = handle_missing_values(df_nsw)



# =========================================
# Step 3: Feature Engineering
# =========================================

# Convert 'Time' to datetime format and extract the hour
df_nsw['Time'] = pd.to_datetime(df_nsw['Time'], format='%H:%M:%S', errors='coerce')
df_nsw['Hour'] = df_nsw['Time'].dt.hour


# Define a function to cluster time into categories
def cluster_hour(hour):
    if 0 <= hour < 6:
        return 'Night'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

# Apply the function to create the 'Time Cluster' column
df_nsw['Time Cluster'] = df_nsw['Hour'].apply(cluster_hour)

# Define a function to cluster age into groups
def cluster_age_group(age):
    if age == 0:
        return 'Unknown'
    elif 0 <= age <= 16:
        return 'Child'
    elif 17 <= age <= 25:
        return 'Young Adult'
    elif 26 <= age <= 39:
        return 'Adult'
    elif 40 <= age <= 64:
        return 'Middle-Aged'
    elif age >= 65:
        return 'Senior'
    else:
        return 'Unknown'

# Apply the clustering function to the Age column
df_nsw['Age Group Cluster'] = df_nsw['Age'].apply(cluster_age_group)

# Define a function to cluster speed limits
def cluster_speed_limit(speed):
    if speed == 0:
        return 'Unknown'
    elif 1 <= speed <= 30:
        return 'Low Speed'
    elif 31 <= speed <= 60:
        return 'Moderate Speed'
    elif 61 <= speed <= 90:
        return 'High Speed'
    elif 91 <= speed <= 110:
        return 'Very High Speed'
    else:
        return 'Unknown'

# Apply the function to create the 'Speed Cluster' column
df_nsw['Speed Cluster'] = df_nsw['Speed Limit'].apply(cluster_speed_limit)

# Convert 'Speed Limit' to numeric, coerce errors to NaN
df_nsw['Speed Limit'] = pd.to_numeric(df_nsw['Speed Limit'], errors='coerce')

# Display the result
print(df_nsw)

# Count incidents by SA4 Name
LGA_counts = df_nsw['National LGA Name 2021'].value_counts()

# Display top 10 areas with the most incidents
print(LGA_counts.head(10))

# Export the cleaned dataset to an Excel file
output_filename = "Cleaned_NSW_Fatalities.xlsx"
df_nsw.to_excel(output_filename, index=False)

# Display a confirmation message
print(f"Cleaned dataset saved as '{output_filename}'")


##========= Done Descriptive Part ==========##
## Preprocessing data for Random forest  : National LGA Name 2021 ##

# Step 1: Calculate unique incidents for each LGA, including 'Unknown'
lga_incident_counts = df_nsw['National LGA Name 2021'].value_counts(dropna=False)

# Step 2: Define risk levels based on unique incident counts, including 'Unknown'
def classify_risk(lga):
    count = lga_incident_counts.get(lga, 0)
    
    # Handle 'Unknown' explicitly
    if lga == 'Unknown' or pd.isna(lga):
        return 'Unknown'
    
    # Classify risk levels based on incident counts
    if count > 100:
        return 'High-Risk'
    elif 51 <= count <= 100:
        return 'Moderate-Risk'
    else:
        return 'Low-Risk'


# Step 3: Apply the risk classification to create the 'Risk_Level' column
df_nsw['Risk_Level'] = df_nsw['National LGA Name 2021'].apply(classify_risk)

# Step 4: Map Risk Levels to numeric values for the model, keeping 'Unknown' as a separate category
risk_mapping = {'Low-Risk': 0, 'Moderate-Risk': 1, 'High-Risk': 2, 'Unknown': -1}
df_nsw['Risk_Level_Mapped'] = df_nsw['Risk_Level'].map(risk_mapping)

# Display the counts for each risk level, including 'Unknown'
print(df_nsw['Risk_Level'].value_counts())

# Save the updated dataset to a new Excel file
output_filename = "Updated_Cleaned_NSW_Fatalities.xlsx"
df_nsw.to_excel(output_filename, index=False)

# Display a confirmation message
print(f"Updated Cleaned dataset saved as '{output_filename}'")


## Random Forest Classifier for Mapped LGA High Risk Areas  ##
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================================
# Step 1: Prepare the Data
# =========================================
# Assuming `df_nsw` is your DataFrame
# Convert categorical columns to numerical with pd.get_dummies()
df_encoded = pd.get_dummies(df_nsw[['Time Cluster', 'Speed Cluster', 'Age Group Cluster']], drop_first=True)

# Include numerical columns
df_encoded['Hour'] = df_nsw['Hour']

# Target column: Risk_Level_Mapped
df_encoded['Risk_Level_Mapped'] = df_nsw['Risk_Level_Mapped']

# =========================================
# Step 2: Split the Data into Features (X) and Target (y)
# =========================================
X = df_encoded.drop('Risk_Level_Mapped', axis=1)  # Features
y = df_encoded['Risk_Level_Mapped']  # Target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# =========================================
# Step 3: Train the Random Forest Classifier
# =========================================
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# =========================================
# Step 4: Make Predictions and Evaluate the Model
# =========================================
y_pred = rf.predict(X_test)

# Accuracy and Classification Report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================================
# Step 5: Export the First Few Trees (Optional)
# =========================================
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz


# Export the first decision tree from the Random Forest
for i in range(3):
    tree = rf.estimators_[0]
    dot_data = export_graphviz(tree,
                           feature_names=X_train.columns,
                           filled=True,
                           max_depth=2,
                           impurity=False, 
                            proportion=True)
    graph = graphviz.Source(dot_data)
    graph.render(f"LGA_Prediction_Tree_{i}", format="png")

# =========================================
#  Step 1: Load the Cleaned Data
# =========================================
# Assuming your dataset is already cleaned and saved as "Updated_Cleaned_NSW_Fatalities.xlsx"
df = pd.read_excel("Updated_Cleaned_NSW_Fatalities.xlsx")

# =========================================
#  Step 2: Prepare Features and Target
# =========================================
# Define the feature columns and target column
feature_columns = ['Dayweek', 'Road User','Age Group Cluster']
target_column = 'Time Cluster'

# Encode categorical features using one-hot encoding
df_encoded = pd.get_dummies(df[feature_columns], drop_first=True)

# Add the target column
df_encoded[target_column] = df[target_column]

# =========================================
#  Step 3: Split the Data into Features (X) and Target (y)
# =========================================
X = df_encoded.drop(target_column, axis=1)  # Features
y = df_encoded[target_column]  # Target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# =========================================
#  Step 4: Train the Random Forest Model
# =========================================
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# =========================================
#  Step 5: Make Predictions and Evaluate the Model
# =========================================
y_pred = rf.predict(X_test)

# Accuracy and Classification Report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================================
#  Step 6: Visualize and Export the First Few Decision Trees
# =========================================
# Export the first 3 decision trees from the Random Forest
for i in range(3):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=X_train.columns,
                               filled=True,
                               max_depth=3,  # Adjust max depth if needed
                               impurity=False,
                               proportion=True)
    graph = graphviz.Source(dot_data)
    graph.render(f"TimeCluster_Prediction_Tree_{i}", format="png")
    print(f"Decision tree {i+1} saved as 'TimeCluster_Prediction_Tree_{i}.png'")

# =========================================
#  Step 7: Save the Model Output to Excel
# =========================================
# Add predictions to the test set and save it to an Excel file
X_test['Actual Hour'] = y_test
X_test['Predicted Hour'] = y_pred

# Export the results
output_filename = "TimeCluster_Prediction_Results.xlsx"
X_test.to_excel(output_filename, index=False)

# Display a confirmation message
print(f"Prediction results saved as '{output_filename}'")

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Create a new interaction term: Dayweek & Road User
df_nsw['Dayweek_RoadUser'] = df_nsw['Dayweek'] + "_" + df_nsw['Road User']

# Convert categorical columns into numerical encoding
X = df_nsw[['Dayweek_RoadUser', 'Age Group Cluster']]
X = pd.get_dummies(X, drop_first=True)  # Convert to numerical

# Encode the target variable (Time Cluster)
y = y.astype('category').cat.codes  # Convert categorical target to numerical

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Classifier
xgb_model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(y.unique()), eval_metric="mlogloss")

# Train Model
xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print("XGBoost Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

