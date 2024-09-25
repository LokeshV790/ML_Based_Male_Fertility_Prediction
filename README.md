
# Machine Learning-Based Male Fertility Prediction

## Overview

This project predicts male fertility potential using machine learning models trained on the VISEM dataset, which includes sperm motility data. The goal is to automate the sperm analysis process, assisting clinicians and fertility specialists in making informed treatment decisions. The project uses Gradient Boosting as the primary algorithm to analyze sperm motility and health parameters.

The web application is built using Flask, allowing users to upload sperm motility videos, input clinical parameters, and receive fertility predictions along with suggestions for improving fertility prognosis.

## Features

- **Data Preprocessing**: Processes and extracts frames from sperm motility videos.
- **Machine Learning Model**: Gradient Boosting model used for fertility prediction.
- **Model Training**: Detailed steps to train the model using video and health data, and save the trained model.
- **Flask Web Application**: An intuitive web interface where users can upload videos and input clinical data.
- **Recommendations**: Provides suggestions based on the fertility prediction.

## Dataset

This project uses the VISEM Dataset, which contains video data of sperm motility and various clinical health parameters.

- Dataset link: [VISEM Dataset](https://dataverse.no/dataset.xhtml?persistentId=doi:10.18710/FKXFKO)
- **Note**: Ensure compliance with the dataset’s licensing and usage guidelines.

## Requirements

To run this project, ensure the following dependencies are installed:

- Python 3.7+
- Required libraries (listed in `requirements.txt`):
  - scikit-learn
  - OpenCV
  - Flask
  - Joblib
  - NumPy
  - Pandas

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/LokeshV790/ML_Based_Male_Fertility_Prediction.git
   ```

2. **Install Dependencies:**

   Navigate to the project directory and install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download and Preprocess the Dataset:**

   Download the VISEM dataset from the link above. Place the videos in the appropriate directory as per the project structure.

4. **Preprocess the Video Data:**

   Run the preprocessing script to extract frames and clean the video data.

   ```bash
   python preprocess_data.py
   ```

5. **Train the Model:**

   Train the Gradient Boosting model using the following command:

   ```bash
   python train_model.py
   ```

6. **Run the Web Application:**

   After training the model, run the Flask web application:

   ```bash
   python app.py
   ```

7. **Access the Web App:**

   Open your web browser and navigate to `http://localhost:5000` to access the application.

## Project Structure

```
ML_Based_Male_Fertility_Prediction/
|
├── app.py                     # Main Flask application file
├── preprocess_data.py          # Script to preprocess video data
├── train_model.py              # Model training script
├── requirements.txt            # List of required dependencies
├── models/                     # Directory containing saved machine learning models
├── templates/                  # Flask HTML templates
├── static/                     # Static files like CSS, images (for Flask)
├── data/                       # Directory for dataset files (videos, CSVs, etc.)
└── README.md                   # Project README file
```

## Model and Algorithm

The Gradient Boosting algorithm was chosen for the prediction of sperm motility and fertility potential. It provided the best balance between accuracy and interpretability for this specific dataset.

| Model               | Mean Squared Error | R-Squared |
|---------------------|--------------------|-----------|
| Gradient Boosting    | 645.67             | 0.73      |

### Best Model

The Gradient Boosting model was selected due to its superior performance in terms of mean squared error and R-squared value.

## Future Enhancements

- Real-time Video Processing: Add functionality to process videos in real-time or stream data from a microscope.
- Enhanced Recommendations: Improve the recommendation engine to suggest more precise treatment options.
- Cloud Deployment: Deploy the application on cloud platforms like AWS, GCP, or Heroku for greater accessibility.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Feel free to raise an issue or submit a pull request.
