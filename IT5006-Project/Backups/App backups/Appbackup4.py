import pandas as pd
import geopandas as gpd
import plotly.express as px
import streamlit as st

# Load Singapore GeoJSON
singapore_towns_geojson = '2-planning-area.geojson'
sg_geojson = gpd.read_file(singapore_towns_geojson)

# Function to load and preprocess data
@st.experimental_memo
def load_and_preprocess_data():
    # Load the resale data
    resale_data = pd.read_csv('resale_data_cleaned.csv', parse_dates=['month'])
    resale_data['town'] = resale_data['town'].str.upper()  # Ensure town names match GeoJSON
    aggregated_data = resale_data.groupby('town')['price_per_sqm'].mean().reset_index()
    return aggregated_data

# Function to create choropleth map
def make_choropleth(input_df, geojson_data):
    custom_color_scale = [
        [0.0, "rgb(0, 255, 255)"],  # Cyan for the lowest prices
        [0.1, "rgb(173, 216, 230)"],  # Light Blue
        [0.2, "rgb(135, 206, 250)"],  # Deep Sky Blue
        [0.3, "rgb(0, 191, 255)"],  # Deep Sky Blue somewhat darker
        [0.4, "rgb(0, 128, 0)"],  # Green
        [0.5, "rgb(173, 255, 47)"],  # Green-yellow
        [0.6, "rgb(255, 255, 0)"],  # Yellow
        [0.7, "rgb(255, 165, 0)"],  # Orange
        [0.8, "rgb(255, 69, 0)"],  # Red-Orange
        [0.9, "rgb(255, 0, 0)"],  # Red
        [1.0, "rgb(139, 0, 0)"]  # Dark Red for the highest prices
    ]
    choropleth = px.choropleth(input_df,
                               geojson=geojson_data,
                               locations='town',
                               featureidkey="properties.name",
                               color='price_per_sqm',
                               color_continuous_scale=custom_color_scale,
                               labels={'price_per_sqm': 'Price per sqm'})
    choropleth.update_geos(fitbounds="locations", visible=False)
    choropleth.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=600)
    # Disable the color legend
    choropleth.update_layout(coloraxis_showscale=False)
    return choropleth

# Assuming your main app logic is correctly set up, add the display of the side panel where appropriate
def main():
    st.title("HDB Price Dashboard")

    # Load and preprocess data (ensure df is loaded as expected)
    df = load_and_preprocess_data()

    col1, col2 = st.columns([3, 1])  # Adjust the ratio as needed

    with col1:
        # Create choropleth map
        sg_map = make_choropleth(df, sg_geojson)
        st.plotly_chart(sg_map, use_container_width=True)

    with col2:
        st.write("Price per sqm by Town")
        # Display the side panel with towns and their color-coded prices
        # You can adjust the presentation here as needed
        for index, row in df.iterrows():
            # Use the background color corresponding to the town's value
            # This example does not compute the exact color; adjust as needed.
            st.markdown(f"**{row['town']}**: {row['price_per_sqm']:.2f} SGD/sqm")

if __name__ == "__main__":
    main()
