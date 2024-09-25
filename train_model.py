import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Step 1: Load the dataset
df = pd.read_csv('/content/data/combined_data_with_videos.csv')

# Step 2: Preprocess data (handling missing values, data cleaning)
def clean_and_fill_data(df):
    df.replace('Not reported', pd.NA, inplace=True)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(df.mean(), inplace=True)
    return df

df = clean_and_fill_data(df)

# Step 3: Define features (X) and target (y)
# Inputs from the user and video features
features = ['Abstinence time(days)', 'Body mass index (kg/m²)', 'Age (years)',
            'Normal spermatozoa (%)', 'Teratozoospermia index', 'Total sperm count (x10⁶)', 
            'Sperm vitality (%)', 'Progressive motility (%)', 'Sperm concentration (x10⁶/mL)',
            'Head defects (%)', 'Tail defects (%)', 'Video Mean Feature', 'Video Std Feature']

# Choose a target to classify as 'Fertile' or 'Not Fertile'
df['Fertile'] = df['Progressive motility (%)'] > 32  # Assuming fertility threshold at 32%
X = df[features]
y = df['Fertile']

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a Gradient Boosting Classifier
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 7: Train on the entire dataset
model.fit(X, y)

# Step 8: Save the model to a file
joblib.dump(model, 'models/gradient_boosting_model.pkl')
print("Model saved as 'models/gradient_boosting_model.pkl'")
