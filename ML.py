import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Load Training Data
train_data = pd.read_excel('Mapping Auto/train.xlsx')

# ----------- Feature Engineering -----------

vectorizer = TfidfVectorizer(stop_words="english")
X_course = vectorizer.fit_transform(train_data["Course"])
kmeans = KMeans(n_clusters=15, random_state=42)
train_data["CourseCluster"] = kmeans.fit_predict(X_course)

# Encode Location
label_encoders = {}

# Extract Seniority from Job Titles
def extract_seniority(role):
    role = str(role).lower()  # Convert to lowercase and handle missing values

    if any(word in role for word in ["ceo", "founder", "chief", "president", "co-founder", "executive"]):
        return "Executive"
    elif any(word in role for word in ["vp", "vice president"]):
        return "VP"
    elif any(word in role for word in ["director", "head", "principal"]):
        return "Director"
    elif any(word in role for word in ["manager", "lead", "supervisor"]):
        return "Manager"
    elif any(word in role for word in ["professor", "lecturer", "adjunct"]):
        return "Professor"
    elif any(word in role for word in ["research scientist", "staff scientist", "scientist", "fellow", "research associate"]):
        return "Scientist"
    elif any(word in role for word in ["researcher", "postdoctoral", "research assistant", "graduate researcher", "phd student"]):
        return "Researcher"
    elif any(word in role for word in ["engineer", "developer", "software"]):
        return "Engineer"
    elif any(word in role for word in ["quantitative researcher", "quant", "analyst", "trader"]):
        return "Quant/Analyst"
    elif any(word in role for word in ["data scientist", "machine learning", "ai", "ml"]):
        return "Data/ML Scientist"
    elif any(word in role for word in ["consultant", "business analyst"]):
        return "Consultant"
    elif any(word in role for word in ["intern", "summer", "assistant"]):
        return "Intern"
    elif "not specified" in role or "na" in role or "error" in role:
        return "Unknown"
    else:
        return "Other"

train_data["Seniority"] = train_data["Role"].apply(extract_seniority)


# Extract Industry from Company Name
def extract_industry(company):
    company = str(company).lower()  # Convert to lowercase and handle missing values

    if any(word in company for word in ["university", "college", "school", "academy", "institute", "research", "lab", "center", "observatory"]):
        return "Academia"
    elif any(word in company for word in ["google", "netflix", "stripe", "microsoft", "facebook", "meta", "amazon", "openai", "deepmind", "waymo", "samsung", "apple", "nvidia", "sony", "ibm", "tesla", "linkedin"]):
        return "Tech"
    elif any(word in company for word in ["goldman sachs", "renaissance", "citadel", "squarepoint", "quant", "trading", "hedge", "capital", "investment", "bank", "finance", "fidelity", "ubs", "d.e. shaw", "optiver", "wellington"]):
        return "Finance"
    elif any(word in company for word in ["nasa", "world bank", "government", "army", "navy", "air force", "darpa", "energy", "us department", "u.s. department", "noaa"]):
        return "Government"
    elif any(word in company for word in ["pharma", "biotech", "health", "medical", "hospital", "therapeutics", "pharmaceuticals", "gsk", "sanofi", "moderna", "pfizer", "illumina", "bristol myers", "genentech", "novartis"]):
        return "Healthcare/Biotech"
    elif any(word in company for word in ["consult", "mckinsey", "bcg", "bain", "deloitte", "pwc", "ey", "bdo"]):
        return "Consulting"
    elif any(word in company for word in ["automotive", "ford", "gm", "toyota", "rivian", "tesla", "aerospace", "boeing", "raytheon", "northrop", "lockheed"]):
        return "Aerospace/Automotive"
    elif any(word in company for word in ["energy", "oil", "shell", "bp", "chevron", "exxon", "electric", "climate", "sustainability"]):
        return "Energy"
    elif "not specified" in company or "na" in company or "error" in company:
        return "Unknown"
    else:
        return "Other"

train_data["Industry"] = train_data["Current Company"].apply(extract_industry)

# Encode Seniority & Industry
for col in ["Seniority", "Industry"]:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le  # Store encoder for test data

# Define Features & Target
features = ["CourseCluster", "Seniority", "Industry"]

X = train_data[features]
y = train_data["Department"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------- Model Training -----------
param_grid = {
    "n_estimators": [50, 100, 200, 300, 400, 500, 700, 1000],
    "max_depth": [10, 15, 20, 25, 30, 35, 50],
    "min_samples_split": [2, 3, 4, 5, 6, 7, 10, 15]
}

clf = RandomForestClassifier(random_state=1)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Train Final Model
best_clf = RandomForestClassifier(**best_params, random_state=1)
best_clf.fit(X_train, y_train)

# Predictions & Evaluation
predictions = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))



# Print unique department names
print("Departments in y_test:", y_test.unique())

# Generate Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)

# Print raw Confusion Matrix
print("\nConfusion Matrix (Raw Numbers):")
print(conf_matrix)

# Visualize Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=y_test.unique(), yticklabels=y_test.unique())
plt.xlabel("Predicted Department")
plt.ylabel("Actual Department")
plt.title("Confusion Matrix")
plt.show()


# ----------- Predicting Departments for Unlabeled Test Data -----------
test_data = pd.read_excel("C:/Users/OliverStanton/Programming/Python/Maven Projects/Mapping Auto/test.xlsx")

# Apply K-Means Clustering to Test Data Courses
X_test_course = vectorizer.transform(test_data["Course"])
test_data["CourseCluster"] = kmeans.predict(X_test_course)

# Encode Test Data Location, Seniority, and Industry
test_data["Seniority"] = test_data["Role"].apply(extract_seniority)
test_data["Seniority"] = label_encoders["Seniority"].transform(test_data["Seniority"])
test_data["Industry"] = test_data["Current Company"].apply(extract_industry)
test_data["Industry"] = label_encoders["Industry"].transform(test_data["Industry"])

# Select Features & Predict
X_final_test = test_data[features]
test_data["PredictedDepartment"] = best_clf.predict(X_final_test)

# Save Predictions
file_name = "Test Predictions 2"
test_data.to_excel(file_name, index=False)
print(f"\nPredictions saved to {file_name}")
