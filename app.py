import streamlit as st
import pandas as pd
import numpy as np

st.title("ML Web Interface Demo")

upload_file = st.file_uploader("Upload a CSV file", type=["csv"])
if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    target_column = st.selectbox("Select the target column", df.columns)
    st.write(f"Selected target column: {target_column}")