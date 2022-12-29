from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext


st.header('Sentiment Analysis')
with st.expander('klasifikasi Text'):
    text = st.text_input('Text here: ')
    if text:
        blob = TextBlob(text)
        st.write('Polarity: ', round(blob.sentiment.polarity,2))
        st.write('Subjectivity: ', round(blob.sentiment.subjectivity,2))
       
with st.expander('Analisis masal data'):
    upl = st.file_uploader('Upload file')

    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

#
    def analyze(x):
        if x >= 0.5:
            return 'Positive'
        elif x <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

#
    if upl:
        df = pd.read_excel(upl)
        del df['Unnamed: 0']
        df['score'] = df['Content'].apply(score)
        df['sentimen'] = df['score'].apply(analyze)
        st.write(df.head(10))

        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )

    if st.button("Predict"):
      # Plot the prediction results
      count = df['sentimen'].value_counts()
      plt.figure(figsize=(10, 6))
      plt.bar(['Positive', 'Netral', 'Negative'], count, color=['royalblue','green', 'orange'])
      plt.xlabel('Jenis Sentimen', size=14)
      plt.ylabel('Frekuensi', size=14)
      st.pyplot()


