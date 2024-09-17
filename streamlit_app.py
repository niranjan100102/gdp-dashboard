# prompt: from above code create web app application using streamlit in git hub

%%writefile app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

# Load the data
df = pd.read_csv('cyber insurance data.csv')

# Assuming 'WILLINGNESS TO BUY THE PRODUCT' is your target variable
# Replace 'WILLINGNESS TO BUY THE PRODUCT' with the actual column name
y = df['WILLINGNESS TO BUY THE PRODUCT']

# Select features for prediction
# Replace 'AGE', 'GENDER', etc. with the actual column names from your dataset
X = df[['AGE', 'CYBER THREAT', 'HAVING INSURANCE']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy:", accuracy)


# Calculate R-squared
r_squared = r2_score(y_test, y_pred)
#print("R-squared:", r_squared)

# Streamlit app
st.title("Cyber Insurance Prediction")

# Input features
age = st.number_input("Age", min_value=0, max_value=120)
cyber_threat = st.selectbox("Cyber Threat", ["Low", "Medium", "High"])
having_insurance = st.selectbox("Having Insurance", ["No", "Yes"])

# Map categorical values to numerical
if cyber_threat == "Low":
  cyber_threat_val = 0
elif cyber_threat == "Medium":
  cyber_threat_val = 1
else:
  cyber_threat_val = 2

if having_insurance == "No":
  having_insurance_val = 0
else:
  having_insurance_val = 1

# Create input array
input_data = [[age, cyber_threat_val, having_insurance_val]]

# Make prediction
prediction = model.predict(input_data)[0]

# Display prediction
if st.button("Predict"):
  st.write("Prediction:", prediction)

# Display model performance
st.write("Model Accuracy:", accuracy)
st.write("R-squared:", r_squared)

