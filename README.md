# SentiFlow - IMDB Movie Review Sentiment Analyzer

A production-ready sentiment analysis web application that classifies movie reviews as positive or negative using machine learning. Built with scikit-learn and Flask, this project demonstrates end-to-end ML workflow from data preprocessing to model deployment.

## 🌟 Features

- **Machine Learning Pipeline**: Complete ML pipeline with data preprocessing, feature engineering, and model training
- **Web Interface**: Flask-based web application for real-time sentiment prediction
- **High Accuracy**: Logistic Regression model trained on IMDB dataset with TF-IDF vectorization
- **Comprehensive Analysis**: Jupyter notebooks documenting the entire development process
- **Modular Architecture**: Clean separation of concerns with dedicated modules for data loading, preprocessing, and evaluation
- **Model Persistence**: Trained model serialization using joblib for quick deployment

## 📁 Project Structure

```
sentimentanalysis/
├── main.py                      # Model training and evaluation pipeline
├── SentimentAnalyzerEngine.py   # Flask web application server
├── awsengine.py                 # AWS deployment configuration
├── requirements.txt             # Project dependencies
├── dataloader/                  # Data loading utilities
├── datapreprocessing/           # Text preprocessing and vectorization
│   ├── datapreprocessing.py
│   └── vectorizer.py
├── evaluation/                  # Model evaluation metrics and plots
├── models/                      # Saved trained models
├── sentimentanalyzer/          # Core analyzer modules
├── templates/                   # HTML templates for web interface
│   ├── home.html
│   └── predict.html
├── notebooks/                   # Jupyter notebooks with detailed analysis
│   ├── Phase 1 Data Preprocessing and EDA
│   ├── Phase 2 Feature Engineering and Feature Selection
│   ├── Phase 3 Model Selection
│   └── Phase 4 Model Evaluation and XAI
└── data/                        # Dataset directory (IMDB-Dataset.csv)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sentimentanalysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (required for lemmatization)
   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

### Training the Model

Run the main training pipeline:

```bash
python main.py
```

This will:
- Load the IMDB dataset
- Preprocess and clean the text data
- Train a Logistic Regression classifier with TF-IDF features
- Evaluate the model and display metrics (Precision, AUC, F1-Score)
- Save the trained model to `models/model/classifier.pkl`

### Running the Web Application

Start the Flask server:

```bash
python SentimentAnalyzerEngine.py
```

The application will be available at `http://localhost:5000`

1. Navigate to the home page
2. Enter a movie review in the text box
3. Click submit to get the sentiment prediction (Positive/Negative)

## 🔧 Technical Details

### Machine Learning Pipeline

The project uses a scikit-learn Pipeline consisting of:

1. **Data Cleaning**: Custom transformer for text preprocessing
   - HTML tag removal
   - Special character cleaning
   - Text normalization

2. **Feature Extraction**: TF-IDF Vectorization
   - Word-level analysis with lemmatization
   - N-gram range: 1-3
   - Max features: 10,000
   - Min document frequency: 10

3. **Classification**: Logistic Regression
   - L2 regularization
   - LBFGS solver
   - Multi-class support

### Model Performance

The model is evaluated using:
- **Precision Score**: Measures prediction accuracy
- **AUC Score**: Area under ROC curve
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions

## 📊 Notebooks

The `notebooks/` directory contains detailed Jupyter notebooks documenting:

- **Phase 1**: Data Preprocessing and Exploratory Data Analysis
- **Phase 2**: Feature Engineering and Feature Selection
- **Phase 3**: Model Selection and Comparison
- **Phase 4**: Model Evaluation and Explainable AI (XAI)

## 🧪 Testing

Run unit tests:

```bash
python unittest_file.py
```

## 🛠️ Technologies Used

- **Python 3.x**: Core programming language
- **scikit-learn**: Machine learning framework
- **Flask**: Web application framework
- **NLTK**: Natural language processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Joblib**: Model serialization
- **BeautifulSoup**: HTML parsing

## 📈 Future Enhancements

- [ ] Add support for multiple languages
- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Deploy to cloud platforms (AWS, Heroku)
- [ ] Add REST API endpoints
- [ ] Implement batch prediction
- [ ] Add model monitoring and retraining pipeline

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

Yash Kumar Gols

## 🙏 Acknowledgments

- IMDB dataset for movie reviews
- scikit-learn community for excellent ML tools
- Flask community for web framework support

---

**Project Name Suggestion**: SentiFlow - A smooth, flowing sentiment analysis experience
