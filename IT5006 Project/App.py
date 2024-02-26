import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import json
import plotly.express as px
import plotly.graph_objects as go
import calendar

st.set_page_config(page_title="IT5006 Project", page_icon=":smiley:", layout="wide")

# Load data
df_rental = pd.read_csv('./Cleaned_data/rental_cleaned.csv', parse_dates=['rent_approval_date'])
df_resale = pd.read_csv('./Cleaned_data/resale_data_cleaned.csv', parse_dates=['month'])
df_resale['flat_type'] = df_resale['flat_type'].str.replace(' ', '-', regex=False) # making categories consistent

# Sidebar + Filters
# st.sidebar.success("Select a page above")
rental_resale_sltn = st.sidebar.radio(
    'HDB Rental or Resale?',
    ("Rental", "Resale")
    )
if rental_resale_sltn == 'Rental':
    min_sltn_value = pd.to_datetime(df_rental['rent_approval_date'].min()).year
    max_sltn_value = pd.to_datetime(df_rental['rent_approval_date'].max()).year
    # Extract unique values for town and flat type filters
    unique_towns = sorted(df_rental['town'].unique())
    unique_flat_types = sorted(df_rental['flat_type'].unique())
else:
    min_sltn_value = pd.to_datetime(df_resale['month'].min()).year
    max_sltn_value = pd.to_datetime(df_resale['month'].max()).year
    # Extract unique values for town and flat type filters
    unique_towns = sorted(df_resale['town'].unique())
    unique_flat_types = sorted(df_resale['flat_type'].unique())

date_sltn = st.sidebar.slider(
    'Select date range',
    min_sltn_value, max_sltn_value, (min_sltn_value,max_sltn_value)
    )

# Sidebar filter for town
st.sidebar.write(' ')
if 'select_all_towns' not in st.session_state:
    st.session_state['select_all_towns'] = True
select_all_towns = st.sidebar.checkbox('Select All Towns', value=st.session_state['select_all_towns'])
st.session_state['select_all_towns'] = select_all_towns
default_selection_towns = unique_towns if select_all_towns else []
selected_town = st.sidebar.multiselect('Or, Select Specific Town(s)', unique_towns, default=default_selection_towns)

# Sidebar filter for flat type
st.sidebar.write(' ')
if 'select_all_flat_types' not in st.session_state:
    st.session_state['select_all_flat_types'] = True
select_all_flat_types = st.sidebar.checkbox('Select All Flat Types', value=st.session_state['select_all_flat_types'])
st.session_state['select_all_flat_types'] = select_all_flat_types
default_selection_flat_types = unique_flat_types if select_all_flat_types else []
selected_flat_type = st.sidebar.multiselect('Or, Select Specific Flat Type(s)', unique_flat_types, default=default_selection_flat_types)

# Header
st.subheader("HDB Data - Exploratory Data Analysis")

# Loading the selected dataset to dataframe based on filters
if rental_resale_sltn == 'Rental':
  df = df_rental.iloc[:, 1:]     # Drop first column
  df = df[(df['rent_approval_date'].dt.year >= date_sltn[0]) & (df['rent_approval_date'].dt.year <= date_sltn[1])]
  # Apply town and flat type filters
  df = df[df['town'].isin(selected_town) & df['flat_type'].isin(selected_flat_type)]
else:
  df = df_resale
  df = df[(df['month'].dt.year >= date_sltn[0]) & (df['month'].dt.year <= date_sltn[1])]
  # Apply town and flat type filters
  df = df[df['town'].isin(selected_town) & df['flat_type'].isin(selected_flat_type)]


# Common visualisations


# Function to load and preprocess data based on user selection (For zach's visualisation)
def load_and_preprocess_data(data_type, selected_year_start, selected_year_end, geojson_data, selected_town, selected_flat_type):
    # Load data based on the selected type
    if data_type == 'Resale':
        data = pd.read_csv('./Cleaned_data/resale_data_cleaned.csv', parse_dates=['month'])
        data['year'] = data['month'].dt.year
    else:
        data = pd.read_csv('./Cleaned_data/rental_cleaned.csv', parse_dates=['rent_approval_date'])
        data['year'] = data['rent_approval_date'].dt.year
        data['price_per_sqm'] = data['price_per_sqm']

    data['town'] = data['town'].str.upper()

    # Filter data for the selected year
    data = data[(data['year'] >= selected_year_start) & (data['year'] <= selected_year_end)]

    # Apply town and flat type filters
    data = data[data['town'].isin(selected_town) & data['flat_type'].isin(selected_flat_type)]

    # Extract town names from GeoJSON
    geojson_towns = [feature['properties']['name'] for feature in geojson_data['features']]
    
    # Filter dataframe to only include towns found in the GeoJSON
    data = data[data['town'].isin(geojson_towns)]

    # Aggregate data
    if data_type == 'Resale':
        aggregated_data = data.groupby('town')['price_per_sqm'].median().reset_index()
    else:
        aggregated_data = data.groupby('town')['price_per_sqm'].median().reset_index()

    # Create a dataframe for towns in GeoJSON but missing in data
    aggregated_data_towns = set(aggregated_data['town'])
    missing_towns = [town for town in geojson_towns if town not in aggregated_data_towns]
    missing_df = pd.DataFrame({'town': missing_towns, 'price_per_sqm': [0] * len(missing_towns)})

    # Combine aggregated data with missing towns
    final_df = pd.concat([aggregated_data, missing_df], ignore_index=True)

    return final_df


# Function to create choropleth map
def make_choropleth(input_df, geojson_data):
    # Define a more granular color scale
    custom_color_scale = [
        [0, "rgb(194,197,204)"],  # White for NaN or values outside the expected range
        [1/27, "rgb(240,249,232)"],  # Lightest green
        [2/27, "rgb(204,235,197)"],
        [3/27, "rgb(168,221,181)"],
        [4/27, "rgb(123,204,196)"],
        [5/27, "rgb(78,179,211)"],
        [6/27, "rgb(43,140,190)"],
        [7/27, "rgb(8,104,172)"],
        [8/27, "rgb(8,64,129)"],  # Blue
        [9/27, "rgb(39,100,125)"],
        [10/27, "rgb(60,130,142)"],
        [11/27, "rgb(85,158,131)"],
        [12/27, "rgb(107,184,104)"],
        [13/27, "rgb(137,204,74)"],  # Green-Yellow
        [14/27, "rgb(192,217,59)"],
        [15/27, "rgb(217,234,43)"],
        [16/27, "rgb(237,248,33)"],  # Yellow
        [17/27, "rgb(252,217,125)"],
        [18/27, "rgb(252,186,3)"],
        [19/27, "rgb(252,153,0)"],  # Orange
        [20/27, "rgb(252,129,0)"],
        [21/27, "rgb(252,104,0)"],
        [22/27, "rgb(252,78,42)"],
        [23/27, "rgb(241,58,19)"],  # Red
        [24/27, "rgb(224,33,0)"],
        [25/27, "rgb(204,12,0)"],
        [26/27, "rgb(180,0,0)"],  # Darker red
        [1.0, "rgb(160,0,0)"]  # Darkest red
    ]
    choropleth = px.choropleth(input_df,
                               geojson=geojson_data,
                               locations='town',
                               featureidkey="properties.name",
                               color='price_per_sqm',
                               color_continuous_scale=custom_color_scale,
                               labels={'price_per_sqm': 'Price per sqm'})
    choropleth.update_geos(fitbounds="locations", visible=False)

    # Disable zoom and pan
    choropleth.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=750, width=650,
        dragmode=False,
        geo=dict(
            projection_scale=5, # Keep this or adjust as needed for initial scale
            center=dict(lat=1.3521, lon=103.8198), # Centering on Singapore, adjust as needed
        )
    )

    return choropleth

# Function to map values to colors based on the custom color scale used for the choropleth
def map_value_to_color(value, min_value, max_value):
    # Define the custom color scale as used in the choropleth
    custom_color_scale = [
        "rgb(255, 255, 255)",  # White for NaN or values outside the expected range
        "rgb(240, 249, 232)",  # Lightest green
        "rgb(204, 235, 197)",
        "rgb(168, 221, 181)",
        "rgb(123, 204, 196)",
        "rgb(78, 179, 211)",
        "rgb(43, 140, 190)",
        "rgb(8, 104, 172)",
        "rgb(8, 64, 129)",  # Blue
        "rgb(39, 100, 125)",
        "rgb(60, 130, 142)",
        "rgb(85, 158, 131)",
        "rgb(107, 184, 104)",
        "rgb(137, 204, 74)",  # Green-Yellow
        "rgb(192, 217, 59)",
        "rgb(217, 234, 43)",
        "rgb(237, 248, 33)",  # Yellow
        "rgb(252, 217, 125)",
        "rgb(252, 186, 3)",
        "rgb(252, 153, 0)",  # Orange
        "rgb(252, 129, 0)",
        "rgb(252, 104, 0)",
        "rgb(252, 78, 42)",
        "rgb(241, 58, 19)",  # Red
        "rgb(224, 33, 0)",
        "rgb(204, 12, 0)",
        "rgb(180, 0, 0)",  # Darker red
        "rgb(160, 0, 0)"   # Darkest red
    ]
    
    # Handle NaN values by returning a default color, e.g., grey
    if pd.isnull(value):
        return "rgb(128, 128, 128)"  # Grey for NaN or missing values
    
    # Normalize the value within the range
    normalized_value = (value - min_value) / (max_value - min_value)
    # Ensure the normalized value is within [0, 1] after handling edge cases
    normalized_value = max(0, min(normalized_value, 1))
    # Map the normalized value to an index in the color scale
    color_index = int(normalized_value * (len(custom_color_scale) - 1))
    return custom_color_scale[color_index]

def plot_additional_graphs(data_type, selected_year_start, selected_year_end,  selected_town, selected_flat_type, data):
    if data_type == 'Resale':
        # Updating df according to filters
        data['town'] = data['town'].str.upper()
        data['year'] = data['month'].dt.year
        data['month_only'] = data['month'].dt.month
        month_names = [calendar.month_abbr[i] for i in range(1, 13)]
        data = data[(data['year'] >= date_sltn[0]) & (data['year'] <= date_sltn[1])]

        # Breakdown by Flat Type (pie chart)
        st.markdown("**Breakdown by Flat Types**")
        flat_type_counts = data['flat_type'].value_counts().sort_index()
        flat_types = flat_type_counts.index.unique() ### want to arrange alphabetically
        fig = go.Figure(data=[go.Pie(labels=flat_types, values=flat_type_counts, direction='clockwise', hole=.3, sort=False)])
        st.plotly_chart(fig, use_container_width=True)

        # Graph 1: Price per sqm against flat type
        st.markdown("**Price per Sqm by Flat Type**")
        plt.figure(figsize=(10, 6))
        data.groupby('flat_type')['price_per_sqm'].median().plot(kind='bar', color='skyblue')
        plt.ylabel('Median Price per Sqm')
        plt.xlabel('Flat Type')
        st.pyplot(plt)
        plt.clf()  # Clear the figure after plotting

        # Graph 2: Price per sqm over the selected year, aggregated by months (if needed)
        # Assuming you want to show variations within the selected year. 
        # This requires a different approach, considering your data spans multiple years.
        
        # If you intended to show changes over the years, consider removing the year filter for this graph.
        st.markdown("**Price per Sqm across the years**")
        plt.figure(figsize=(10, 6))
        # Ensure there's data to plot
        if not data.empty:
            data.groupby(data['month'].dt.strftime('%Y-%m'))['price_per_sqm'].median().plot(kind='line', color='green')
            #plt.title('Price per Sqm Over Months between ' + str(selected_year_start) + ' and ' + str(selected_year_end) )
            plt.ylabel('Median Price per Sqm')
            plt.xlabel('Month')
            st.pyplot(plt)
        else:
            st.write("No data available for this year.")
        plt.clf()  # Clear the figure after plotting
        
        # Median resale price by year, trend line showing how prices changed over the years
        median_resale_by_year_flat_type = data.groupby(['year', 'flat_type'])['price_per_sqm'].median().reset_index()
            # Create a line chart with Plotly Express
        fig = px.line(median_resale_by_year_flat_type, x='year', y='price_per_sqm', color='flat_type',
                    title='Median Resale Price per sqm by Year and Flat Type',
                    labels={'price_per_sqm': 'Median Resale Price per sqm', 'year': 'Year'},
                    markers=True)
            # Add a trend line for overall median rental price by year
        overall_median_resale_by_year = data.groupby('year')['price_per_sqm'].median().reset_index()
        fig.add_scatter(x=overall_median_resale_by_year['year'], y=overall_median_resale_by_year['price_per_sqm'],
                        mode='lines+markers', name='Overall Trend', line=dict(color='black', dash='dash'))
        fig.update_xaxes(tickmode='linear', dtick=1)
        st.plotly_chart(fig, use_container_width=True)

        # Median resale price by month, trend line showing whether certain months are peak
        median_resale_by_month_flat_type = data.groupby(['month_only', 'flat_type'])['price_per_sqm'].median().reset_index()

        fig = px.line(median_resale_by_month_flat_type, x='month_only', y='price_per_sqm', color='flat_type',
                    title='Median Resale Price per sqm by Month and Flat Type',
                    labels={'price_per_sqm': 'Median Resale Price per sqm', 'month_only': 'Month'},
                    markers=True)

        overall_median_rental_by_month = data.groupby('month_only')['price_per_sqm'].median().reset_index()
        fig.add_scatter(x=overall_median_rental_by_month['month_only'], y=overall_median_rental_by_month['price_per_sqm'],
                        mode='lines+markers', name='Overall Trend', line=dict(color='black', dash='dash'))
        fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=month_names)
        st.plotly_chart(fig, use_container_width=True)

        # Graph 3: Number of flats sold against towns
        st.markdown("**Number of Flats Sold by Town**")
        plt.figure(figsize=(10, 6))
        data['town'].value_counts().plot(kind='bar', color='orange')
        plt.ylabel('Number of Flats Sold')
        plt.xlabel('Town')
        st.pyplot(plt)
        plt.clf()  # Clear the figure after plotting
        
        # Graph 4: Number of flats sold against flat type
        st.markdown("**Number of Flats Sold by Flat Type**")
        plt.figure(figsize=(10, 6))
        data['flat_type'].value_counts().plot(kind='bar', color='teal')
        plt.ylabel('Number of Flats Sold')
        plt.xlabel('Flat Type')
        st.pyplot(plt)
        plt.clf()  # Clear the figure after plotting
        
        # Number of units resold by year
        units_by_year = data['year'].value_counts().sort_index().reset_index()
        units_by_year.columns = ['Year', 'Number of Units Resold']

        fig = px.bar(units_by_year, x='Year', y='Number of Units Resold', title='Number of Units Resold by Year',
                     labels={'Year': "Year", 'Number of Units Resold': "Number of Units Resold"})

        fig.update_layout(bargap=0.5)
        fig.update_xaxes(tickmode='linear', dtick=1)
        st.plotly_chart(fig, use_container_width=True)        

        # Number of units resold by month Trend line showing the number of units rented over the months (i.e. whether there are peaks at certain months)
        data['month_only'] = data['month'].dt.month
        df_selected_year = data[(data['year'] >= date_sltn[0]) & (data['year'] <= date_sltn[1])]
        units_resold_by_month_flat_type = df_selected_year.groupby(['month_only', 'flat_type']).size().reset_index(name='count')
        fig = px.line(units_resold_by_month_flat_type, x='month_only', y='count', color='flat_type',
                title=f'Number of Units Resold by Month',
                labels={'count': 'Number of Units Resold', 'month_only': 'Month'},
                markers=True)
        overall_units_rented_by_month = df_selected_year.groupby('month_only')['month'].count().reset_index(name='count')
        fig.add_scatter(x=overall_units_rented_by_month['month_only'], y=overall_units_rented_by_month['count'],
                        mode='lines+markers', name='Overall Trend', line=dict(color='black', dash='dash'))
        fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=month_names)
        st.plotly_chart(fig, use_container_width=True)

        # Towns ranked by median price by town
        median_price_by_town = data.groupby('town')['price_per_sqm'].median().sort_values(ascending=False)
        top_10_median_price_by_town = median_price_by_town

        top_10_towns = top_10_median_price_by_town.index.unique()

        fig = px.bar(x=top_10_towns, y=top_10_median_price_by_town, title='Towns ranked by Median Resale Price per sqm', 
                    labels={'x': "Town", 'y': "Median Price per sqm"})
        st.plotly_chart(fig, use_container_width=True)

        # Resale price vs Distance to MRT
        fig = px.scatter(data, x='mrt_dist', y='price_per_sqm', title='Relationship between Resale Price per sqm & Distance to MRT',
                         labels={'mrt_dist': "Distance to MRT", 'price_per_sqm': "Price per sqm"}, 
                         color='town', hover_data=['town', 'flat_type', 'resale_price'])
        st.plotly_chart(fig, use_container_width=True)

        # Resale price vs flat age
        fig = px.scatter(data, x='flat_age_at_sale', y='price_per_sqm', title='Relationship between Resale Price per sqm & Flat Age',
                         labels={'flat_age_at_sale': "Flat Age", 'price_per_sqm': "Price per sqm"}, 
                         color='town', hover_data=['town', 'flat_type', 'resale_price'])
        st.plotly_chart(fig, use_container_width=True)
        
    
    # Plot For rental data
    else:
        # Updating df according to filters
        data['town'] = data['town'].str.upper()
        data['year'] = data['rent_approval_date'].dt.year
        data['month'] = data['rent_approval_date'].dt.month
        month_names = [calendar.month_abbr[i] for i in range(1, 13)]
        data = data[(data['year'] >= date_sltn[0]) & (data['year'] <= date_sltn[1])]

        # Breakdown by Flat Type (pie chart)
        st.markdown("**Breakdown by Flat Types**")
        flat_type_counts = data['flat_type'].value_counts().sort_index()
        flat_types = flat_type_counts.index.unique() 

        fig = go.Figure(data=[go.Pie(labels=flat_types, values=flat_type_counts, direction='clockwise', hole=.3, sort=False)])

        st.plotly_chart(fig, use_container_width=True)

        # Median rental price by year, trend line showing how rental prices changed over the years
        median_rental_by_year_flat_type = data.groupby(['year', 'flat_type'])['price_per_sqm'].median().reset_index()
            # Create a line chart with Plotly Express
        fig = px.line(median_rental_by_year_flat_type, x='year', y='price_per_sqm', color='flat_type',
                    title='Median Rental Price per sqm by Year and Flat Type',
                    labels={'price_per_sqm': 'Median Rental Price per sqm', 'year': 'Year'},
                    markers=True)
            # Add a trend line for overall median rental price by year
        overall_median_rental_by_year = data.groupby('year')['price_per_sqm'].median().reset_index()
        fig.add_scatter(x=overall_median_rental_by_year['year'], y=overall_median_rental_by_year['price_per_sqm'],
                        mode='lines+markers', name='Overall Trend', line=dict(color='black', dash='dash'))
        fig.update_xaxes(tickmode='linear', dtick=1)
        st.plotly_chart(fig, use_container_width=True)

        # Median rental price by month, trend line showing whether certain months are peak
        median_rental_by_month_flat_type = data.groupby(['month', 'flat_type'])['price_per_sqm'].median().reset_index()

        fig = px.line(median_rental_by_month_flat_type, x='month', y='price_per_sqm', color='flat_type',
                    title='Median Rental Price per sqm by Month and Flat Type',
                    labels={'price_per_sqm': 'Median Rental Price per sqm', 'month': 'Month'},
                    markers=True)

        overall_median_rental_by_month = data.groupby('month')['price_per_sqm'].median().reset_index()
        fig.add_scatter(x=overall_median_rental_by_month['month'], y=overall_median_rental_by_month['price_per_sqm'],
                        mode='lines+markers', name='Overall Trend', line=dict(color='black', dash='dash'))
        fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=month_names)
        st.plotly_chart(fig, use_container_width=True)

        # Number of units rented by year
        units_by_year = data['year'].value_counts().sort_index().reset_index()
        units_by_year.columns = ['Year', 'Number of Units Rented']

        fig = px.bar(units_by_year, x='Year', y='Number of Units Rented', title='Number of Units Rented by Year',
                     labels={'Year': "Year", 'Number of Units Rented': "Number of Units Rented"})

        fig.update_layout(bargap=0.5)
        fig.update_xaxes(tickmode='linear', dtick=1)
        st.plotly_chart(fig, use_container_width=True)        

        # Number of units rented by month Trend line showing the number of units rented over the months (i.e. whether there are peaks at certain months)
        data['month'] = data['rent_approval_date'].dt.month
        df_selected_year = data[(data['year'] >= date_sltn[0]) & (data['year'] <= date_sltn[1])]
        units_rented_by_month_flat_type = df_selected_year.groupby(['month', 'flat_type']).size().reset_index(name='count')
        fig = px.line(units_rented_by_month_flat_type, x='month', y='count', color='flat_type',
                title=f'Number of Units Rented by Month',
                labels={'count': 'Number of Units Rented', 'month': 'Month'},
                markers=True)
        overall_units_rented_by_month = df_selected_year.groupby('month')['rent_approval_date'].count().reset_index(name='count')
        fig.add_scatter(x=overall_units_rented_by_month['month'], y=overall_units_rented_by_month['count'],
                        mode='lines+markers', name='Overall Trend', line=dict(color='black', dash='dash'))
        fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=month_names)
        st.plotly_chart(fig, use_container_width=True)

        # Towns ranked by median rental by town
        median_price_by_town = data.groupby('town')['price_per_sqm'].median().sort_values(ascending=False)
        top_10_median_price_by_town = median_price_by_town

        top_10_towns = top_10_median_price_by_town.index.unique()

        fig = px.bar(x=top_10_towns, y=top_10_median_price_by_town, title='Towns ranked by Median Rental Price per sqm', 
                    labels={'x': "Town", 'y': "Median Price per sqm"})
        st.plotly_chart(fig, use_container_width=True)

        # Rental price vs Distance to MRT
        fig = px.scatter(data, x='mrt_dist', y='price_per_sqm', title='Relationship between Rental Price per sqm & Distance to MRT',
                         labels={'mrt_dist': "Distance to MRT", 'price_per_sqm': "Price per sqm"}, 
                         color='town', hover_data=['town', 'flat_type', 'monthly_rent'])
        st.plotly_chart(fig, use_container_width=True)

        # Rental price vs flat age
        fig = px.scatter(data, x='flat_age', y='price_per_sqm', title='Relationship between Rental Price per sqm & Flat Age',
                         labels={'flat_age': "Flat Age", 'price_per_sqm': "Price per sqm"}, 
                         color='town', hover_data=['town', 'flat_type', 'monthly_rent'])
        st.plotly_chart(fig, use_container_width=True)

## TO DO: insert common visualisations here


# Additional visualisations for resale
if rental_resale_sltn == 'Resale':
  # TO DO: Show additional visualisations
  st.title("Dashboard For Resale Flats ")

  # Overview of numbers
  st.write('### _Overview of Data_')
  median_price = df['resale_price'].median()
  min_price = df['resale_price'].min()
  max_price = df['resale_price'].max()
  no_units = len(df)

  col1, col2, col3 = st.columns(3)
  col1.metric("Median Price", f'S${median_price:,.0f}')
  col2.metric("Min Price", f'S${min_price:,.0f}')
  col3.metric("Max Price", f'S${max_price:,.0f}')

  median_price_per_sqm = df['price_per_sqm'].median()
  min_price_per_sqm = df['price_per_sqm'].min()
  max_price_per_sqm = df['price_per_sqm'].max()
  
  col1, col2, col3 = st.columns(3)
  col1.metric("Median Price per sqm", f'S${median_price_per_sqm:,.0f}')
  col2.metric("Min Price per sqm", f'S${min_price_per_sqm:,.0f}')
  col3.metric("Max Price per sqm", f'S${max_price_per_sqm:,.0f}')

  col1, col2, col3 = st.columns(3)
  col1.metric("No of units", f'{no_units:,}')
  st.write('### _Data Analysis_')

  # Load GeoJSON file
  geojson_file = "map4.json"
  with open(geojson_file) as f:
      sg_geojson = json.load(f)

  # Data Loading and Processing with GeoJSON towns considered
  processed_df = load_and_preprocess_data(rental_resale_sltn, date_sltn[0], date_sltn[1], sg_geojson, selected_town, selected_flat_type)

  st.markdown("**Median Resale Price per sqm by Town**")
  sg_map = make_choropleth(processed_df, sg_geojson)
  st.plotly_chart(sg_map, use_container_width=False)

  # Plot additional graphs if data type is 'Resale'
  plot_additional_graphs(rental_resale_sltn, date_sltn[0], date_sltn[1], selected_town, selected_flat_type, df)

# For rental data
else:
    st.title("Dashboard For Rental Flats ")

    # Overview of numbers
    st.write('### _Overview of Data_')
    median_price = df['monthly_rent'].median()
    min_price = df['monthly_rent'].min()
    max_price = df['monthly_rent'].max()
    no_units = len(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Median Price", f'S${median_price:,.0f}')
    col2.metric("Min Price", f'S${min_price:,.0f}')
    col3.metric("Max Price", f'S${max_price:,.0f}')

    median_price_per_sqm = df['price_per_sqm'].median()
    min_price_per_sqm = df['price_per_sqm'].min()
    max_price_per_sqm = df['price_per_sqm'].max()

    col1, col2, col3 = st.columns(3)
    col1.metric("Median Price per sqm", f'S${median_price_per_sqm:,.2f}')
    col2.metric("Min Price per sqm", f'S${min_price_per_sqm:,.2f}')
    col3.metric("Max Price per sqm", f'S${max_price_per_sqm:,.2f}')

    col1, col2, col3 = st.columns(3)
    col1.metric("No of units", f'{no_units:,}')

    st.write('### _Data Analysis_')
    
    # Load GeoJSON file
    geojson_file = "map4.json"
    with open(geojson_file) as f:
        sg_geojson = json.load(f)

    # Data Loading and Processing with GeoJSON towns considered
    processed_df = load_and_preprocess_data(rental_resale_sltn, date_sltn[0], date_sltn[1], sg_geojson, selected_town, selected_flat_type)

    st.markdown("**Median Rental Price per sqm by Town**")
    sg_map = make_choropleth(processed_df, sg_geojson)
    st.plotly_chart(sg_map, use_container_width=False)
    # Plot additional graphs if data type is 'Rental'
    plot_additional_graphs(rental_resale_sltn, date_sltn[0], date_sltn[1], selected_town, selected_flat_type, df)
