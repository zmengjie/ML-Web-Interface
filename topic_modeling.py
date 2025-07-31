# topic_modeling.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn_model as sklearn_lda_vis
import streamlit.components.v1 as components

def topic_modeling_ui():
    st.header("🧠 Topic Modeling")

    uploaded_file = st.file_uploader("Upload a CSV with a text column", type="csv")
    if not uploaded_file:
        return

    df = pd.read_csv(uploaded_file)
    text_columns = df.select_dtypes(include="object").columns.tolist()
    if not text_columns:
        st.warning("No text columns found.")
        return

    col = st.selectbox("Select text column", text_columns)
    num_topics = st.slider("Number of topics", 2, 15, 5)

    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df[col].fillna(""))

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    st.subheader("Top Words per Topic")
    words = vectorizer.get_feature_names_out()
    for i, topic in enumerate(lda.components_):
        st.markdown(f"**Topic {i+1}:** " + ", ".join([words[i] for i in topic.argsort()[-10:][::-1]]))

    # pyLDAvis
    with st.spinner("Generating interactive visualization..."):
        vis_html = sklearn_lda_vis.prepare(lda, X, vectorizer)
        html = pyLDAvis.prepared_data_to_html(vis_html)
        components.html(html, height=800, scrolling=True)
