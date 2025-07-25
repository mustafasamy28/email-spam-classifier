# 📧 Email Spam Classifier

A professional machine learning web application for classifying emails as spam or legitimate (ham) using Natural Language Processing and Support Vector Machine algorithms.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **Real-time Classification**: Instantly classify emails as spam or ham
- **High Accuracy**: Achieves 98.92% accuracy on test data
- **Professional UI**: Clean, modern Streamlit interface
- **Feature Engineering**: Uses TF-IDF vectorization with message length and punctuation analysis
- **Model Persistence**: Automatic model saving and loading
- **Batch Processing**: Upload CSV/TSV files for training
- **Confidence Scoring**: Shows prediction confidence levels
- **Example Testing**: Pre-loaded spam/ham examples for quick testing

## 🚀 Demo

![Demo Animation](assets/Demo)

*Live demo of the email spam classifier in action*

## 📸 Screenshots

<div align="center">
  <img src="assets/1" width="45%" alt="Main Interface" />
  <img src="assets/2" width="45%" alt="Model Training" />
</div>

<div align="center">
  <img src="assets/3" width="45%" alt="Spam Detection Result" />
  <img src="assets/4" width="45%" alt="Ham Classification Result" />
</div>

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn (Linear SVM)
- **NLP**: TF-IDF Vectorization
- **Data Processing**: Pandas, NumPy
- **Language**: Python 3.8+

## 📁 Project Structure

```
email-spam-classifier/
│
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── spam_classifier_app.py       # Main Streamlit application
├── model_training.ipynb         # Jupyter notebook for model development
├── .gitignore                   # Git ignore file
├── LICENSE                      # MIT License
│
├── data/                        # Dataset directory
│   ├── smsspamcollection.tsv   # Training dataset
│   └── README.md               # Data description
│
├── models/                      # Trained models directory
│   ├── spam_model.pkl          # Saved trained model
│   └── model_metrics.json      # Model performance metrics
│
├── assets/                      # Images and documentation assets
│   ├── Demo                     # Demo animation/video
│   ├── 1                        # Main interface screenshot
│   ├── 2                        # Model training screenshot
│   ├── 3                        # Spam detection result
│   └── 4                        # Ham classification result
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb    # Data analysis
│   ├── 02_feature_engineering.ipynb # Feature creation
│   └── 03_model_comparison.ipynb   # Model evaluation
│
├── src/                         # Source code modules
│   ├── __init__.py
│   ├── data_processor.py       # Data preprocessing functions
│   ├── model_trainer.py        # Model training utilities
│   └── utils.py                # Helper functions
│
└── tests/                       # Unit tests
    ├── __init__.py
    ├── test_classifier.py      # Classifier tests
    └── test_preprocessing.py   # Preprocessing tests
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/mustafasamy28/email-spam-classifier.git
   cd email-spam-classifier
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run spam_classifier_app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## 📊 Usage

### Training the Model

1. **Prepare your data**: Ensure your CSV/TSV file has columns: `label`, `message`
2. **Upload data**: Use the sidebar file upload feature
3. **Train model**: Click "🚀 Train Model" button
4. **View results**: Check accuracy and model statistics

### Making Predictions

1. **Enter email text** in the main text area
2. **Click "🔍 Classify Email"** to get prediction
3. **View results**: See spam/ham classification with confidence score
4. **Try examples**: Use pre-loaded examples for quick testing

### API Usage (Optional)

```python
from src.model_trainer import SpamClassifier

# Initialize classifier
classifier = SpamClassifier()

# Load trained model
classifier.load_model('models/spam_model.pkl')

# Make prediction
email_text = "Congratulations! You've won $1000000!"
prediction, confidence = classifier.predict(email_text)

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2f}")
```

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 98.92% |
| Precision | 98.85% |
| Recall | 98.92% |
| F1-Score | 98.88% |

### Feature Importance

- **TF-IDF Features**: 85% importance
- **Message Length**: 10% importance  
- **Punctuation Count**: 5% importance

## 🔍 Model Architecture

```
Input Email Text
       ↓
Text Preprocessing
       ↓
Feature Extraction
   ↓        ↓
TF-IDF    Numerical Features
Vectors   (Length, Punctuation)
       ↓
   Feature Union
       ↓
Linear SVM Classifier
       ↓
Spam/Ham Prediction
```

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Run specific tests:

```bash
python -m pytest tests/test_classifier.py -v
```

## 📁 Dataset

The project uses the SMS Spam Collection Dataset:
- **Total samples**: 5,572 messages
- **Spam messages**: 747 (13.4%)
- **Ham messages**: 4,825 (86.6%)
- **Source**: UCI Machine Learning Repository

## 🚀 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with one click

### Docker

```bash
# Build image
docker build -t spam-classifier .

# Run container
docker run -p 8501:8501 spam-classifier
```

### Heroku

```bash
# Login and create app
heroku login
heroku create your-spam-classifier

# Deploy
git push heroku main
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Commit changes**: `git commit -am 'Add new feature'`
4. **Push to branch**: `git push origin feature/new-feature`
5. **Submit pull request**

### Development Guidelines

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Ensure all tests pass

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Mostafa Samy**
- GitHub: [@mustafasamy28](https://github.com/mustafasamy28)
- LinkedIn: [Mostafa Samy](https://linkedin.com/in/mostafa-samy)
- Email: mostafasamy28@gmail.com

## 🙏 Acknowledgments

- SMS Spam Collection Dataset from UCI ML Repository
- Streamlit team for the amazing framework
- scikit-learn contributors
- Open source community

## 📞 Support

If you encounter any issues or have questions:

1. **Check existing issues**: [GitHub Issues](https://github.com/mustafasamy28/email-spam-classifier/issues)
2. **Create new issue**: Provide detailed description and steps to reproduce
3. **Email support**: mostafasamy28@gmail.com

## 🔮 Future Enhancements

- [ ] **Deep Learning Models**: Integration with BERT/RoBERTa
- [ ] **Multi-language Support**: Spam detection in multiple languages
- [ ] **Email Integration**: Direct Gmail/Outlook API integration
- [ ] **Advanced Analytics**: Word clouds and feature visualization
- [ ] **A/B Testing**: Model comparison interface
- [ ] **Real-time Learning**: Online learning capabilities

---

⭐ **Star this repository if you found it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/mustafasamy28/email-spam-classifier.svg?style=social&label=Star)](https://github.com/mustafasamy28/email-spam-classifier)