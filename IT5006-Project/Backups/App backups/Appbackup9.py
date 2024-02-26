import pandas as pd
import geopandas as gpd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt

# Load Singapore GeoJSON
singapore_towns_geojson = '2-planning-area.geojson'
sg_geojson = gpd.read_file(singapore_towns_geojson)

# Function to load and preprocess data based on user selection
@st.cache
def load_and_preprocess_data(data_type, selected_year):
    if data_type == 'Resale':
        # Load the resale data
        data = pd.read_csv('resale_data_cleaned.csv', parse_dates=['month'])
        data['year'] = data['month'].dt.year  # Extract year for filtering
    else:
        # Load the rental data
        data = pd.read_csv('rental_data_cleaned.csv', parse_dates=['rent_approval_date'])
        data['year'] = data['rent_approval_date'].dt.year  # Extract year for filtering
        data['price_per_sqm'] = data['monthly_rent']  # Assuming monthly rent as price per sqm for consistency in visualization

    data['town'] = data['town'].str.upper()  # Ensure town names match GeoJSON
    data = data[data['year'] == selected_year]  # Filter data by selected year

    if data_type == 'Resale':
        aggregated_data = data.groupby('town')['price_per_sqm'].mean().reset_index()
    else:
        aggregated_data = data.groupby('town')['monthly_rent'].mean().reset_index()
        aggregated_data.rename(columns={'monthly_rent': 'price_per_sqm'}, inplace=True)

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
    return choropleth

# Function to map values to colors based on the custom color scale used for the choropleth
def map_value_to_color(value, min_value, max_value):
    # Define the custom color scale as used in the choropleth
    custom_color_scale = [
        "rgb(0, 255, 255)",  # Cyan
        "rgb(173, 216, 230)",  # Light Blue
        "rgb(135, 206, 250)",  # Deep Sky Blue
        "rgb(0, 191, 255)",  # Deep Sky Blue somewhat darker
        "rgb(0, 128, 0)",  # Green
        "rgb(173, 255, 47)",  # Green-yellow
        "rgb(255, 255, 0)",  # Yellow
        "rgb(255, 165, 0)",  # Orange
        "rgb(255, 69, 0)",  # Red-Orange
        "rgb(255, 0, 0)",  # Red
        "rgb(139, 0, 0)"  # Dark Red
    ]
    # Normalize the value within the range
    normalized_value = (value - min_value) / (max_value - min_value)
    # Map the normalized value to an index in the color scale
    color_index = int(normalized_value * (len(custom_color_scale) - 1))
    return custom_color_scale[color_index]

# Function to display the side panel with town names, their price per sqm, and color coding
def display_side_panel(df, col2):
    # Assuming df is already sorted by price_per_sqm in descending order
    # Calculate min and max price per sqm for progress bar scaling
    min_price = df['price_per_sqm'].min()
    max_price = df['price_per_sqm'].max()

    # Configure the dataframe display inside the specified column (col2)
    with col2:
        st.markdown('#### Towns & Price per Sqm')
        
        st.dataframe(df,
                     column_order=("town", "price_per_sqm"),
                     hide_index=True,
                     width=None,
                     column_config={
                        "town": st.column_config.TextColumn("Town"),
                        "price_per_sqm": st.column_config.ProgressColumn(
                            "Price per Sqm",
                            format="%.2f",  # Assuming you want to format the numbers as floats with two decimal places
                            min_value=min_price,
                            max_value=max_price,
                         )}
                     )



def plot_additional_graphs(data_type, selected_year):
    if data_type == 'Resale':
        data = pd.read_csv('resale_data_cleaned.csv', parse_dates=['month'])
        data['year'] = data['month'].dt.year
        data['town'] = data['town'].str.upper()
        data = data[data['year'] == selected_year]
        
        # Graph 1: Price per sqm against flat type
        plt.figure(figsize=(10, 6))
        data.groupby('flat_type')['price_per_sqm'].mean().plot(kind='bar', color='skyblue')
        plt.title('Price per Sqm by Flat Type')
        plt.ylabel('Average Price per Sqm')
        plt.xlabel('Flat Type')
        st.pyplot(plt)
        plt.clf()  # Clear the figure after plotting
        
        # Graph 2: Price per sqm over the selected year, aggregated by months (if needed)
        # Assuming you want to show variations within the selected year. 
        # This requires a different approach, considering your data spans multiple years.
        
        # If you intended to show changes over the years, consider removing the year filter for this graph.
        plt.figure(figsize=(10, 6))
        # Ensure there's data to plot
        if not data.empty:
            data.groupby(data['month'].dt.strftime('%Y-%m'))['price_per_sqm'].mean().plot(kind='line', color='green')
            plt.title('Price per Sqm Over Months in ' + str(selected_year))
            plt.ylabel('Average Price per Sqm')
            plt.xlabel('Month')
            st.pyplot(plt)
        else:
            st.write("No data available for this year.")
        plt.clf()  # Clear the figure after plotting
        
        # Graph 3: Number of flats sold against towns
        plt.figure(figsize=(10, 6))
        data['town'].value_counts().plot(kind='bar', color='orange')
        plt.title('Number of Flats Sold by Town')
        plt.ylabel('Number of Flats Sold')
        plt.xlabel('Town')
        st.pyplot(plt)
        plt.clf()  # Clear the figure after plotting
        
        # Graph 4: Number of flats sold against flat type
        plt.figure(figsize=(10, 6))
        data['flat_type'].value_counts().plot(kind='bar', color='teal')
        plt.title('Number of Flats Sold by Flat Type')
        plt.ylabel('Number of Flats Sold')
        plt.xlabel('Flat Type')
        st.pyplot(plt)
        plt.clf()  # Clear the figure after plotting


def main():
    # Setup: Title, Sidebar Inputs
    st.title("HDB Price Dashboard")
    data_type, selected_year = setup_sidebar()

    # Data Loading and Processing
    df = load_and_preprocess_data(data_type, selected_year)

    # Layout Configuration: Columns for Map and Side Panel
    col1, col2 = st.columns([3, 1])

    with col1:
        # Ensure make_choropleth is called once and its result is used here
        sg_map = make_choropleth(df, sg_geojson)
        st.plotly_chart(sg_map, use_container_width=True)
        
        # Additional Graphs (if applicable)
        if data_type == 'Resale':
            plot_additional_graphs(data_type, selected_year)

    with col2:
        # Call the updated display_side_panel function
        display_side_panel(df, col2)


def setup_sidebar():
    # Sidebar configuration and return selected options
    data_type = st.sidebar.radio("Select Data Type", ("Resale", "Rental"))
    years = list(range(1990, 2024))
    selected_year = st.sidebar.selectbox("Select Year", years)
    return data_type, selected_year

# Assume other functions (load_and_preprocess_data, make_choropleth, display_side_panel, plot_additional_graphs) are defined elsewhere

if __name__ == "__main__":
    main()
