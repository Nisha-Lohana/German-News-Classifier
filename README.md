# German News Articles Classifier

## Project Overview

This project is a machine learning-based system for automatic classification of German news articles into predefined categories. It supports both text input and voice input. In voice input, speech is converted into text and then classified using trained machine learning models.

The system uses Natural Language Processing (NLP) techniques along with TF-IDF feature extraction and multiple machine learning algorithms. Different models are trained and compared to identify the best performing classifier for German news classification.

---

## Dataset Information

- **Total Articles:** 9245
- **Total Categories:** 9

### Category Distribution

- Panorama: 1510
- Web: 1509
- International: 1360
- Wirtschaft: 1270
- Sport: 1081
- Inland: 913
- Etat: 601
- Wissenschaft: 516
- Kultur: 485

---

## Features

- Classification of German news articles into multiple categories
- Text input prediction
- Voice input using speech-to-text conversion
- Comparison of multiple machine learning models
- Streamlit web application
- Saved trained models and vectorizer
- Performance evaluation using charts

---

## Screenshots

### Streamlit Application

![Streamlit Application](images/streamlit.png)

---

### German News Classification

![German News Classification](images/german_news_classification.png)

---

### Text Input Interface

![Text Input](images/text_input.png)

---

### Text Input Results

![Text Input Results](images/text_input_results.png)

---

### Voice Input Results

![Voice Input Results](images/voice_input_results.png)

---

### Final Results

![Results](images/Results.png)

---

### Accuracy Comparison Chart

![Accuracy Chart](images/accuracy_chart.png)

---

### F1 Score Comparison Chart

![F1 Score Chart](images/f1_chart.png)

---

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)
---

## Project Structure

```text
GERMAN PROJECT/

├── app.py
├── train_model.py
├── generate_all_charts.py
├── Articles.csv
├── tfidf.pkl
├── all_models.pkl
├── models_report.txt
├── accuracy_chart.png
├── f1_chart.png
├── confusion_matrix.png
├── article_audio.wav
├── new.py
└── README.md
```

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- SpeechRecognition
- TF-IDF Vectorizer
- Machine Learning Algorithms
- Matplotlib

---

## Machine Learning Models Used

- Logistic Regression
- Naive Bayes (MultinomialNB)
- Linear Support Vector Classifier (LinearSVC)
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Random Forest
- Gradient Boosting
- AdaBoost
- Extra Trees Classifier

---

## Feature Engineering

### Text Preprocessing

- Lowercasing
- URL removal
- Removal of numbers and special characters
- Removal of German stopwords

### Feature Extraction

- TF-IDF Vectorization
- Unigrams and Bigrams (1–2 n-grams)

---

## Model Training Process

1. Load dataset
2. Clean and preprocess text
3. Convert text into TF-IDF features
4. Split dataset into training and testing sets
5. Train multiple machine learning models
6. Evaluate models using Accuracy, Precision, Recall, and F1 Score
7. Save trained models and vectorizer

---

## Model Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

Detailed results are stored in **models_report.txt**.

---

## Visualization (Charts)

The following charts are generated:

- Accuracy Comparison Chart
- F1 Score Comparison Chart
- Confusion Matrix (Best Model)

---

## How to Run the Project

```bash
pip install -r requirements.txt
python train_model.py
python generate_all_charts.py
streamlit run app.py
```

---

## Voice Input Feature

- Supports WAV audio files
- Uses Google Speech Recognition API
- Converts German speech into text
- Classifies converted text using trained models

---

## Conclusion

This project demonstrates the use of Natural Language Processing and Machine Learning for automatic classification of German news articles. It compares multiple models and uses evaluation metrics along with visualizations to identify the best performing classifier.

---

## Future Improvements

- Integration of Deep Learning models such as LSTM or BERT
- Real-time news classification
- Improved UI/UX dashboard
- Deployment on cloud platforms

---

## Note

Due to GitHub's file size limit, trained model files (`all_models.pkl` and `tfidf.pkl`) are not included in this repository.

---

## Author

**Nisha Lohana**

Computer Science Graduate | Web Developer | UI/UX Designer | 3D Modeller
