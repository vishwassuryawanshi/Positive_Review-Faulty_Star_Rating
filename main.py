import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textblob
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import re

nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

port = PorterStemmer()

def text_cleaner(text):
    cleaned = re.sub('[^a-zA-Z]', " ", text)
    cleaned = cleaned.lower()
    cleaned = cleaned.split()
    cleaned = [port.stem(word) for word in cleaned if word not in stopwords.words("english")]
    cleaned = ' '.join(cleaned)
    return cleaned

st.title("Identifying Review's Rating")
st.header("Instructions")
st.markdown("1.Review column's name should be **Text**")
st.markdown("2.Rating column's name should be **Star**")
st.markdown("3.Rating range should be 0-5")

uploaded_file = st.file_uploader("Choose a File")

df = pd.read_csv(uploaded_file)
st.write(df)

if st.button("Click for Results") :
    df["Cleaned_Text"] = df["Text"].apply(lambda x: text_cleaner(str(x)))

    sid = SentimentIntensityAnalyzer()

    df["Vader_Score"] = df["Cleaned_Text"].apply(lambda review:sid.polarity_scores(review))
    df["Vader_Compound_Score"]  = df['Vader_Score'].apply(lambda score_dict: score_dict['compound'])
    df["Result"] = df["Vader_Compound_Score"].apply(lambda c: 'positive' if c > 0 else ('negative' if c < 0 else 'neutral'))
    st.bar_chart(df.Result.value_counts())

    df_positive = df[(df.Result == "positive")]
    df_positive["Opinion_Positive"] = df_positive["Star"].apply(lambda star: "No Attention Needed" if star >= 3 else "Attention Needed")
    st.bar_chart(df_positive.Opinion_Positive.value_counts())

    data = df_positive

    st.download_button(
        label="Download data as CSV",
        data=data.to_csv().encode("utf-8"),
        file_name='data.csv',
        mime='text/csv',
    )
