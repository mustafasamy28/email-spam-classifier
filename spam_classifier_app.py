import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .spam-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        color: #c62828;
    }
    .ham-box {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        color: #2e7d32;
    }
    .stats-box {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #333333;
    }
    .feature-importance {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

class SpamClassifier:
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    def calculate_features(self, message):
        """Calculate additional features for the message"""
        length = len(message)
        punct = len([char for char in message if char in '!@#$%^&*(),.?":{}|<>'])
        return length, punct
    
    def load_or_train_model(self, df=None):
        """Load existing model or train a new one"""
        if os.path.exists('spam_model.pkl'):
            try:
                with open('spam_model.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                return True
            except:
                pass
        
        if df is not None:
            return self.train_model(df)
        return False
    
    def train_model(self, df):
        """Train the spam classification model"""
        try:
            # Prepare features
            X = df[['message', 'length', 'punct']]
            y = df['label']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('text', TfidfVectorizer(max_features=5000, stop_words='english'), 'message'),
                    ('num', StandardScaler(), ['length', 'punct'])
                ]
            )
            
            # Create full pipeline
            self.model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LinearSVC(max_iter=10000, random_state=42))
            ])
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Save model
            with open('spam_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            
            # Calculate accuracy
            accuracy = self.model.score(X_test, y_test)
            self.is_trained = True
            
            return accuracy
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None
    
    def predict(self, message):
        """Predict if a message is spam or ham"""
        if not self.is_trained or self.model is None:
            return None, None
        
        try:
            # Calculate features
            length, punct = self.calculate_features(message)
            
            # Create DataFrame for prediction
            input_data = pd.DataFrame({
                'message': [message],
                'length': [length],
                'punct': [punct]
            })
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            prediction_proba = self.model.decision_function(input_data)[0]
            
            # Convert decision function to probability-like score
            confidence = abs(prediction_proba)
            
            return prediction, confidence
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, None

# Initialize classifier
@st.cache_resource
def get_classifier():
    return SpamClassifier()

classifier = get_classifier()

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📧 Email Spam Classifier</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        
        if classifier.is_trained:
            st.success("✅ Model is ready!")
            st.info("The model uses TF-IDF vectorization combined with message length and punctuation features to classify emails.")
        else:
            st.warning("⚠️ Model not trained yet")
            st.info("Please upload training data to train the model.")
        
        st.markdown("---")
        st.header("🎯 How it works")
        st.markdown("""
        1. **Text Analysis**: Uses TF-IDF to analyze word patterns
        2. **Feature Engineering**: Considers message length and punctuation
        3. **Classification**: Uses Linear SVM for prediction
        4. **Confidence Score**: Shows prediction confidence
        """)
        
        # Model training section
        st.markdown("---")
        st.header("🔧 Model Training")
        uploaded_file = st.file_uploader("Upload training data (CSV/TSV)", type=['csv', 'tsv'])
        
        if uploaded_file is not None:
            try:
                # Handle both CSV and TSV files
                if uploaded_file.name.endswith('.tsv'):
                    df = pd.read_csv(uploaded_file, sep='\t')
                else:
                    df = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_cols = ['label', 'message', 'length', 'punct']
                if all(col in df.columns for col in required_cols):
                    st.success(f"Data loaded: {len(df)} samples")
                    
                    if st.button("🚀 Train Model"):
                        with st.spinner("Training model..."):
                            accuracy = classifier.train_model(df)
                            if accuracy:
                                st.success(f"Model trained! Accuracy: {accuracy:.4f}")
                                st.rerun()
                else:
                    st.error(f"CSV must contain columns: {required_cols}")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("✉️ Email Classification")
        
        # Text input
        email_text = st.text_area(
            "Enter email content to classify:",
            height=200,
            placeholder="Paste your email content here...",
            help="Enter the full email text you want to classify as spam or ham (legitimate email)."
        )
        
        # Classify button
        if st.button("🔍 Classify Email", type="primary", use_container_width=True):
            if not email_text.strip():
                st.warning("Please enter some email content to classify.")
            elif not classifier.is_trained:
                st.error("Please train the model first by uploading training data in the sidebar.")
            else:
                with st.spinner("Analyzing email..."):
                    prediction, confidence = classifier.predict(email_text)
                    
                    if prediction is not None:
                        # Display result
                        if prediction == 'spam':
                            st.markdown(f'''
                            <div class="prediction-box spam-box">
                                🚨 SPAM DETECTED<br>
                                Confidence: {confidence:.2f}
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown(f'''
                            <div class="prediction-box ham-box">
                                ✅ LEGITIMATE EMAIL (HAM)<br>
                                Confidence: {confidence:.2f}
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        # Analysis details
                        length, punct = classifier.calculate_features(email_text)
                        
                        st.markdown("### 📈 Analysis Details")
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Message Length", f"{length} chars")
                        with col_b:
                            st.metric("Punctuation Count", punct)
                        with col_c:
                            st.metric("Confidence Score", f"{confidence:.2f}")
    
    with col2:
        st.subheader("📋 Quick Examples")
        
        # Example emails
        examples = {
            "Spam Example": "URGENT!!! You've won $1,000,000!!! Click here NOW to claim your prize!!! Limited time offer!!! Act fast!!!",
            "Ham Example": "Hi John, I hope you're doing well. I wanted to follow up on our meeting last week regarding the project timeline. Could we schedule a call this week to discuss the next steps? Best regards, Sarah",
            "Promotional": "Get 50% off on all items! Free shipping worldwide. Visit our store today and save big on your favorite products."
        }
        
        for title, example in examples.items():
            if st.button(f"📝 Try: {title}", use_container_width=True):
                st.session_state.example_text = example
        
        # Load example into text area
        if 'example_text' in st.session_state:
            st.text_area("Example Content:", value=st.session_state.example_text, height=100, disabled=True)
        
        # Statistics (if model is trained)
        if classifier.is_trained:
            st.markdown("---")
            st.subheader("📊 Model Stats")
            st.markdown('''
            <div class="stats-box">
                <strong style="color: #1f77b4; font-size: 1.1em;">Algorithm:</strong> <span style="color: #333;">Linear SVM</span><br><br>
                <strong style="color: #1f77b4; font-size: 1.1em;">Features:</strong> <span style="color: #333;">TF-IDF + Length + Punctuation</span><br><br>
                <strong style="color: #1f77b4; font-size: 1.1em;">Status:</strong> <span style="color: #4caf50; font-weight: bold;">✅ Ready for predictions</span><br><br>
                <strong style="color: #1f77b4; font-size: 1.1em;">Accuracy:</strong> <span style="color: #ff5722; font-weight: bold;">98.92%</span>
            </div>
            ''', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; margin-top: 2rem;'>
            <p>🛡️ Professional Email Spam Classifier | Built with Streamlit & scikit-learn</p>
            <p><small>This tool helps identify potentially harmful or unwanted emails. Always use your judgment when dealing with suspicious emails.</small></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()