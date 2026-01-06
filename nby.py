import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Page config
st.set_page_config(
    page_title="📱 SMS Spam Classifier",
    page_icon="📱",
    layout="wide"
)

st.title("📱 SMS Spam Classifier - Naive Bayes")
st.markdown("**Upload CSV with `Category` (ham/spam) & `Message` columns**")

# File uploader
csv_file = st.file_uploader(
    "📁 Upload spam dataset (CSV)",
    type=['csv'],
    help="Columns: Category (ham/spam), Message"
)

if csv_file is not None:
    # Load data
    spam_df = pd.read_csv(csv_file)
    spam_df['Category'] = spam_df['Category'].map({'ham': 0, 'spam': 1}).fillna(spam_df['Category'])

    st.success(f"✅ **Dataset loaded:** {len(spam_df):,} messages")

    stat1, stat2 = st.columns(2)
    stat1.metric("✅ Ham", (spam_df['Category'] == 0).sum())
    stat2.metric("📱 Spam", (spam_df['Category'] == 1).sum())

    # Data preview
    st.subheader("📋 Sample Messages")
    st.dataframe(spam_df.head())

    # Train model
    @st.cache_resource
    def build_spam_model(input_df):
        temp_df = input_df.copy()
        temp_df['Message'] = temp_df['Message'].astype(str).str.lower()

        text_vectorizer = CountVectorizer(stop_words='english', max_features=5000)
        feature_matrix = text_vectorizer.fit_transform(temp_df['Message'])
        target_labels = temp_df['Category']

        X_train,_
