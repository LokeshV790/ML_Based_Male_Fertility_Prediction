from flask import Flask, request, render_template
import numpy as np
import joblib
import os
from extract_video_features import extract_video_features

app = Flask(__name__)

# Load the trained model
model = joblib.load("models/gradient_boosting_model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video_file' not in request.files:
        return "No video file uploaded", 400

    # Save the uploaded video
    video_file = request.files['video_file']
    video_path = os.path.join('uploads', video_file.filename)
    video_file.save(video_path)

    # Extract video features (6 features)
    (progressive_motility_feature, sperm_concentration_feature, 
     head_defects_feature, tail_defects_feature, 
     video_mean_feature, video_std_feature) = extract_video_features(video_path)

    # Input fields for other features with default handling
    abstinence_time = float(request.form.get('abstinence_time', 0))
    bmi = float(request.form.get('bmi', 0))
    age = int(request.form.get('age', 0))
    normal_sperm = float(request.form.get('normal_sperm', 0))  # Handle missing 'normal_sperm'
    teratozoospermia_index = float(request.form.get('teratozoospermia_index', 0))
    total_sperm_count = float(request.form.get('total_sperm_count', 0))
    sperm_vitality = float(request.form.get('sperm_vitality', 0))

    # Prepare feature vector with all 13 features (7 user inputs + 6 extracted)
    input_features = np.array([[abstinence_time, bmi, age, normal_sperm, 
                                teratozoospermia_index, total_sperm_count, sperm_vitality, 
                                progressive_motility_feature, sperm_concentration_feature, 
                                head_defects_feature, tail_defects_feature, 
                                video_mean_feature, video_std_feature]])

    # Predict fertility status
    prediction = model.predict(input_features)

    return f"Predicted Fertility Status: {'Fertile' if prediction[0] == 1 else 'Infertile'}"

if __name__ == '__main__':
    app.run(debug=True)
