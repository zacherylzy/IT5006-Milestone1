import streamlit as st
import pandas as pd

st.set_page_config(page_title="IT5006 Project", page_icon=":smiley:", layout="wide")

# Load data
df_rental = pd.read_csv('./Cleaned_data/rental_cleaned.csv', parse_dates=['rent_approval_date'])
df_resale = pd.read_csv('./Cleaned_data/resale_data_cleaned.csv', parse_dates=['month'])

# Sidebar + Filters
# st.sidebar.success("Select a page above")
rental_resale_sltn = st.sidebar.radio(
    'HDB Rental or Resale?',
    ("Rental", "Resale")
    )
if rental_resale_sltn == 'Rental':
    min_sltn_value = pd.to_datetime(df_rental['rent_approval_date'].min()).year
    max_sltn_value = pd.to_datetime(df_rental['rent_approval_date'].max()).year
else: 
    min_sltn_value = pd.to_datetime(df_resale['month'].min()).year
    max_sltn_value = pd.to_datetime(df_resale['month'].max()).year

date_sltn = st.sidebar.slider(
    'Select date range',
    min_sltn_value, max_sltn_value, (min_sltn_value,max_sltn_value)
    )

# Header
st.subheader("Welcome to IT5006 Project")
st.title("HDB Data - Exploratory Data Analysis")

# Loading the selected dataset to dataframe based on filters
if rental_resale_sltn == 'Rental':
  df = df_rental.iloc[:, 1:]     # Drop first column
  df = df[(df['rent_approval_date'].dt.year >= date_sltn[0]) & (df['rent_approval_date'].dt.year <= date_sltn[1])]
else:
  df = df_resale
  df = df[(df['month'].dt.year >= date_sltn[0]) & (df['month'].dt.year <= date_sltn[1])]


# Common visualisations

## Display details of dataset
st.write('### Summary of Data')
if rental_resale_sltn == 'Rental':
  median_price = df['monthly_rent'].median()
  min_price = df['monthly_rent'].min()
  max_price = df['monthly_rent'].max()
else: 
  median_price = df['resale_price'].median()
  min_price = df['resale_price'].min()
  max_price = df['resale_price'].max()
no_units = len(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Median price", f'S${median_price:,.0f}')
col2.metric("Min price", f'S${min_price:,.0f}')
col3.metric("Max price", f'S${max_price:,.0f}')
col4.metric("No of units", f'{no_units:,}')

st.write('### Data Analysis')

## TO DO: insert common visualisations here

# Additional visualisations for resale
if rental_resale_sltn == 'Resale':
  # TO DO: Show additional visualisations
  pass

# Display dataset at bottom of page
st.write('### Data')
st.write(df)

_ = '''
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
    # df.head()
'''
