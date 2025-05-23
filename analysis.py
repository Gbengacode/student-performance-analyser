# analysis.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io


def display_summary(df: pd.DataFrame):
    st.write("### Descriptive Statistics")
    st.dataframe(df.describe())

    st.write("### Dataset Info")
    buffer = io.StringIO()  # ✅ file-like buffer
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)


def display_visuals(df: pd.DataFrame):
    # Correlation heatmap
    st.write("#### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Pairplot
    st.write("#### Pairwise Plot")
    fig = sns.pairplot(df)
    st.pyplot(fig)

    # Distribution of final score
    st.write("#### Distribution of Final Scores")
    fig, ax = plt.subplots()
    sns.histplot(df["final_score"], kde=True, ax=ax)
    st.pyplot(fig)
