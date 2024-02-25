import pandas as pd
import streamlit as st
import plotly.express as px
import geopandas as gpd

# Load Singapore GeoJSON
singapore_towns_geojson = '2-planning-area.geojson'  # Update this path if necessary
sg_geojson = gpd.read_file(singapore_towns_geojson)

# Function to load data (Updated with new caching mechanism)
@st.experimental_memo
def load_data():
    # Load the resale data
    resale_data = pd.read_csv('resale_data_cleaned.csv', parse_dates=['month'])  # Ensure date column is parsed
    # It's important not to mutate this data outside this function to avoid CachedObjectMutationWarning
    return resale_data

# Function to create choropleth map
def make_choropleth(input_df, geojson_data):
    # Ensure the 'town' column in your DataFrame matches the 'name' property in your GeoJSON
    choropleth = px.choropleth(input_df,
                               geojson=geojson_data,
                               locations='town',  # DataFrame column with town names
                               featureidkey="properties.name",  # Path to town names in GeoJSON properties
                               color='price',  # DataFrame column for coloring
                               color_continuous_scale="Viridis",
                               scope="asia",  # Consider adjusting or removing this for better fit
                               labels={'price': 'Average Price'})
    choropleth.update_geos(fitbounds="locations", visible=False)
    choropleth.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=600
    )
    return choropleth


# Main app function with updated date handling
def main():
    st.title("Singapore Real Estate Dashboard")

    # Load and display data
    df = load_data()

    # Assuming dataset_type logic remains unchanged
    dataset_type = st.radio("Choose dataset type:", options=["resale", "rental"])
    df_filtered = df  # Adjust according to dataset_type if necessary

    # Convert 'month' to datetime and ensure it's normalized (date only for consistency)
    df['month'] = pd.to_datetime(df['month']).dt.normalize()

    # Simplified Date Range Selection with corrected frequency 'ME'
    min_date, max_date = df_filtered['month'].min(), df_filtered['month'].max()
    date_options = pd.date_range(min_date, max_date, freq='MS').normalize()
    start_date, end_date = st.select_slider("Select date range:", options=date_options, value=(min_date, max_date))

    # Filtering based on selected date range
    df_filtered = df_filtered[(df_filtered['month'] >= start_date) & (df_filtered['month'] <= end_date)]

    # Proceed with the rest of your logic for displaying the map or other components

if __name__ == "__main__":
    main()
