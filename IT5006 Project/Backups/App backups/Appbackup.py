import streamlit as st
import pandas as pd

st.set_page_config(page_title="IT5006 Project", page_icon=":smiley:", layout="wide")


df = pd.read_csv('resale_data_cleaned.csv')

# Header
st.subheader("Welcome to IT5006 Project")
st.title("Data Analysis")
st.sidebar.success("Select a page above")
st.write("This is a simple example of a Streamlit app.")
st.write("[Learn More](https://docs.streamlit.io/en/stable/)")


with st.container():
  st.write("---")
  left_column, right_column = st.columns(2)
  with left_column:
    st.header("Exploartory Data Analysis")
    st.write("##")
    st.write(
      """
      content
      """
    )
  with right_column:
    df.head()
