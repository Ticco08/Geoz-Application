import streamlit as st
import pandas as pd
import folium
from folium.plugins import Draw, MousePosition
from streamlit_folium import st_folium
from shapely.geometry import Point, box, Polygon
from pyproj import Proj, Transformer
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import base64

# Function to convert easting/northing to lat/lon
@st.cache_data(show_spinner=False)
def convert_easting_northing_to_lat_lon(easting, northing, crs, zone):
    proj_utm = Proj(proj=crs, zone=zone, ellps='WGS84')
    proj_latlon = Proj(proj='latlong', datum='WGS84')
    transformer = Transformer.from_proj(proj_utm, proj_latlon)
    lon, lat = transformer.transform(easting, northing)
    return lat, lon

# Vectorized version of coordinate conversion
@st.cache_data(show_spinner=False)
def convert_coords(df, easting_col, northing_col, crs, zone):
    proj_utm = Proj(proj=crs, zone=zone, ellps='WGS84')
    proj_latlon = Proj(proj='latlong', datum='WGS84')
    transformer = Transformer.from_proj(proj_utm, proj_latlon)
    eastings = df[easting_col].values
    northings = df[northing_col].values
    lons, lats = transformer.transform(eastings, northings)
    return lats, lons

# Function to load data from an Excel sheet
@st.cache_data(show_spinner=False)
def load_excel_sheet(uploaded_file, sheet_name, header_row):
    xls = pd.ExcelFile(uploaded_file, engine='openpyxl')
    df = pd.read_excel(xls, sheet_name=sheet_name, engine='openpyxl', header=header_row)
    df.columns = df.columns.map(str).str.strip()  # Ensure column names are clean
    return df

# Function to add markers to the map
def add_marker_to_map(m, lat, lon, location_id, point_type, color='blue', inside_rectangle=False):
    folium.Marker(
        location=[lat, lon],
        popup=f"ID: {location_id}<br>Type: {point_type}" + ("<br>Inside Selection" if inside_rectangle else "<br>Outside Selection"),
        tooltip=location_id,
        icon=folium.Icon(color=color)
    ).add_to(m)

# Function to plot the map with data
def plot_map(data, easting_col, northing_col, type_col, location_id_col, crs, zone, selected_location_ids=None, marker_color='blue'):
    lats, lons = convert_coords(data, easting_col, northing_col, crs, zone)
    center_lat, center_lon = np.mean(lats), np.mean(lons)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="openstreetmap")

    folium.TileLayer(
        'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Esri World Imagery',
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='Map data: &copy; OpenStreetMap contributors, SRTM | Map style: &copy; OpenTopoMap (CC-BY-SA)',
        name='OpenTopoMap',
        overlay=True,
        control=True
    ).add_to(m)

    folium.TileLayer(
        'https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png',
        attr='Map data: &copy; OpenSeaMap contributors',
        name='OpenSeaMap',
        overlay=True,
        control=True
    ).add_to(m)

    for idx, row in data.iterrows():
        lat, lon = convert_easting_northing_to_lat_lon(row[easting_col], row[northing_col], crs, zone)
        inside_rectangle = selected_location_ids is not None and row[location_id_col] in selected_location_ids
        color = 'green' if inside_rectangle else marker_color
        add_marker_to_map(m, lat, lon, row[location_id_col], row[type_col], color, inside_rectangle)

    folium.LayerControl().add_to(m)

    formatter = "function(num) {return L.Util.formatNum(num, 5);};"
    MousePosition(
        position="topright",
        separator=" Long: ",
        empty_string="NaN",
        lng_first=False,
        num_digits=20,
        prefix="Lat:",
        lat_formatter=formatter,
        lng_formatter=formatter,
    ).add_to(m)

    draw = Draw(
        draw_options={
            "polyline": False,
            "polygon": True,
            "circle": True,
            "rectangle": True,
            "circlemarker": False,
            "marker": False,
        },
        edit_options={"edit": True}
    )
    draw.add_to(m)

    return m

# Function to make column names unique
def make_column_names_unique(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

# Function to load all CPT data
@st.cache_data(show_spinner=False)
def load_all_cpts_data(file, cpt_id_col, depth_col, qnet_col):
    df = load_excel_sheet(file, 'All_CPTs', None)
    if df is not None:
        df = df.drop(index=list(range(0, 24))).reset_index(drop=True)
        df.columns = df.iloc[0]
        df = df.drop(index=[0, 1, 2]).reset_index(drop=True)
        df = make_column_names_unique(df)
        selected_cols = [cpt_id_col, depth_col, qnet_col, 'Zone']
        df = df[selected_cols]
        df[qnet_col] = df[qnet_col].apply(lambda x: x * 1000 if x < 10 else x)
    return df

# Function to load all lab data
@st.cache_data(show_spinner=False)
def load_all_lab_data(file, depth_col, location_id_col, columns):
    df = load_excel_sheet(file, 'All_Lab', 4)
    if df is not None:
        df = make_column_names_unique(df)
        selected_cols = [depth_col, location_id_col, 'Zone'] + columns
        df = df[selected_cols]
    return df

# Function to clean data
def check_and_clean_data(df, columns):
    for col in columns:
        df[col] = df[col].replace('None', 0)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

# Function to interpolate and smooth curve
def interpolate_and_smooth_curve(depths, values, fine_depths):
    valid = ~np.isnan(values)
    interp_func = interp1d(depths[valid], values[valid], kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_values = interp_func(fine_depths)
    smooth_values = smooth_curve(interp_values)
    return smooth_values

# Function to smooth curve
def smooth_curve(curve, window_length=11, polyorder=2):
    valid = ~np.isnan(curve)
    if np.sum(valid) < window_length:  # Ensure there are enough points to apply smoothing
        window_length = np.sum(valid) - 1 if np.sum(valid) % 2 == 0 else np.sum(valid)
    return savgol_filter(curve, window_length, polyorder, mode='nearest')

# Function to calculate percentiles
def calculate_percentiles(df, depth_col, su_col, depths, percentiles):
    p5_values = []
    p50_values = []
    p95_values = []
    for depth in depths:
        su_values = []
        for cpt_id in df['CPT ID'].unique():
            df_cpt = df[df['CPT ID'] == cpt_id]
            if df_cpt[depth_col].min() <= depth <= df_cpt[depth_col].max():
                su_values.append(np.interp(depth, df_cpt[depth_col], df_cpt[su_col]))
        if su_values:
            p5_values.append(np.percentile(su_values, percentiles[0]))
            p50_values.append(np.percentile(su_values, percentiles[1]))
            p95_values.append(np.percentile(su_values, percentiles[2]))
        else:
            p5_values.append(np.nan)
            p50_values.append(np.nan)
            p95_values.append(np.nan)
    return np.array(p5_values), np.array(p50_values), np.array(p95_values)

# Function to plot Su vs Depth for each zone with P5, P50, and P95 calculations
def plot_su_vs_depth_by_zone(df_cpt, cpt_id_col, depth_col, qnet_col, location_ids, nk_low, nk_high, discretization, show_cpts, show_percentiles, df_lab=None, lab_depth_col=None, lab_columns=None):
    zones = df_cpt['Zone'].unique()
    zones_with_data = [zone for zone in zones if not df_cpt[(df_cpt['Zone'] == zone) & (df_cpt[cpt_id_col].isin(location_ids))].empty]

    symbols = ['o', 's', 'D', '^', 'v', 'p', '*', 'h', 'H', 'd', '+']

    for zone in zones_with_data:
        fig, ax = plt.subplots(figsize=(17, 20))
        ax2 = ax.twiny()

        df_zone_cpt = df_cpt[df_cpt['Zone'] == zone]

        # Compute Su values for the entire zone
        df_zone_cpt['Su_low'] = df_zone_cpt[qnet_col] / nk_low
        df_zone_cpt['Su_high'] = df_zone_cpt[qnet_col] / nk_high

        # Remove rows with non-positive Su values
        df_zone_cpt = df_zone_cpt[(df_zone_cpt['Su_low'] > 0) & (df_zone_cpt['Su_high'] > 0)]

        # Define consistent depth intervals
        min_depth = df_zone_cpt[depth_col].min()
        max_depth = df_zone_cpt[depth_col].max()
        new_depths = np.arange(min_depth, max_depth, discretization)

        # Calculate P5, P50, and P95 percentiles for low and high Su values at each depth
        percentiles = [5, 50, 95]
        p5_low, p50_low, p95_low = calculate_percentiles(df_zone_cpt, depth_col, 'Su_low', new_depths, percentiles)
        p5_high, p50_high, p95_high = calculate_percentiles(df_zone_cpt, depth_col, 'Su_high', new_depths, percentiles)

        lowest_p5 = np.minimum(p5_low, p5_high)
        median_p50 = (p50_low + p50_high) / 2
        highest_p95 = np.maximum(p95_low, p95_high)

        # Define the range of valid depths
        valid_depths = new_depths[~np.isnan(lowest_p5) & ~np.isnan(median_p50) & ~np.isnan(highest_p95)]
        min_valid_depth = valid_depths.min() if len(valid_depths) > 0 else np.nan
        max_valid_depth = valid_depths.max() if len(valid_depths) > 0 else np.nan

        if not np.isnan(min_valid_depth) and not np.isnan(max_valid_depth):
            fine_depths = np.linspace(min_valid_depth, max_valid_depth, 200)
            lowest_p5 = interpolate_and_smooth_curve(new_depths, lowest_p5, fine_depths)
            median_p50 = interpolate_and_smooth_curve(new_depths, median_p50, fine_depths)
            highest_p95 = interpolate_and_smooth_curve(new_depths, highest_p95, fine_depths)
        else:
            fine_depths = new_depths

        # Plot individual Su values for each CPT
        if show_cpts:
            for cpt_id in df_zone_cpt[cpt_id_col].unique():
                df_cpt_specific = df_zone_cpt[df_zone_cpt[cpt_id_col] == cpt_id]
                ax.plot(df_cpt_specific['Su_low'], df_cpt_specific[depth_col], label=f'{cpt_id_col}: {cpt_id} (Low)', marker='.', linestyle='-', color='grey', alpha=0.5)
                ax.plot(df_cpt_specific['Su_high'], df_cpt_specific[depth_col], label=f'{cpt_id_col}: {cpt_id} (High)', marker='.', linestyle='--', color='lightgrey', alpha=0.5)

        # Plot P5, P50, and P95 percentiles
        if show_percentiles:
            ax.plot(lowest_p5, fine_depths, label='Lowest P5', color='blue', linestyle='-.', linewidth=2)
            ax.plot(median_p50, fine_depths, label='Median P50', color='green', linestyle='-', linewidth=2)
            ax.plot(highest_p95, fine_depths, label='Highest P95', color='red', linestyle='-', linewidth=2)

        # Include lab tests comparison if specified
        if df_lab is not None:
            df_zone_lab = df_lab[df_lab['Zone'] == zone]
            df_zone_lab_selected = df_zone_lab[df_zone_lab['Location_ID'].isin(location_ids)]
            for i, col in enumerate(lab_columns):
                if col in df_zone_lab_selected.columns:
                    lab_data = df_zone_lab_selected[df_zone_lab_selected[col] > 0]  # Exclude 0 kPa values
                    ax.scatter(lab_data[col], lab_data[lab_depth_col], label=f'Lab Test {col}', marker=symbols[i % len(symbols)], alpha=0.7)

        ax.invert_yaxis()
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'Zone: {zone}')
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xlabel('Su (kPa)')
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', bbox_to_anchor=(1.2, 1), fontsize='small')
        ax.grid(True)

        st.pyplot(fig)

# Function to download data as CSV
def download_data(dataframe, filename):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# Function to plot additional lab tests
def plot_additional_lab_tests(df_lab, depth_col_lab, selected_lab_columns):
    zones = df_lab['Zone'].unique()

    for zone in zones:
        st.subheader(f'Zone: {zone}')
        df_zone_lab = df_lab[df_lab['Zone'] == zone]

        # Plot 'SUW_Measured_kN/m3' and 'SUW_Calculated_kN_m3' on the same plot
        if 'SUW_Measured_kN/m3' in selected_lab_columns and 'SUW_Calculated_kN_m3' in selected_lab_columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            suw_measured_data = df_zone_lab[['SUW_Measured_kN/m3', depth_col_lab]].dropna()
            suw_measured_data = suw_measured_data[suw_measured_data['SUW_Measured_kN/m3'] > 0]  # Exclude 0 values
            suw_calculated_data = df_zone_lab[['SUW_Calculated_kN_m3', depth_col_lab]].dropna()
            suw_calculated_data = suw_calculated_data[suw_calculated_data['SUW_Calculated_kN_m3'] > 0]  # Exclude 0 values
            ax.scatter(suw_measured_data['SUW_Measured_kN/m3'], suw_measured_data[depth_col_lab], label='SUW Measured', marker='o')
            ax.scatter(suw_calculated_data['SUW_Calculated_kN_m3'], suw_calculated_data[depth_col_lab], label='SUW Calculated', marker='x')
            ax.invert_yaxis()
            ax.set_ylabel('Depth (m)')
            ax.set_title('SUW Measured vs Calculated vs Depth')
            ax.set_xlabel('SUW (kN/m3)')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

        # Plot 'UW_Measured_kN/m3' and 'UW_Calculated_kN/m3' on the same plot
        if 'UW_Measured_kN_m3' in selected_lab_columns and 'UW_Calculated_kN_m3' in selected_lab_columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            uw_measured_data = df_zone_lab[['UW_Measured_kN_m3', depth_col_lab]].dropna()
            uw_measured_data = uw_measured_data[uw_measured_data['UW_Measured_kN_m3'] > 0]  # Exclude 0 values
            uw_calculated_data = df_zone_lab[['UW_Calculated_kN_m3', depth_col_lab]].dropna()
            uw_calculated_data = uw_calculated_data[uw_calculated_data['UW_Calculated_kN_m3'] > 0]  # Exclude 0 values
            ax.scatter(uw_measured_data['UW_Measured_kN_m3'], uw_measured_data[depth_col_lab], label='UW Measured', marker='o')
            ax.scatter(uw_calculated_data['UW_Calculated_kN_m3'], uw_calculated_data[depth_col_lab], label='UW Calculated', marker='x')
            ax.invert_yaxis()
            ax.set_ylabel('Depth (m)')
            ax.set_title('UW Measured vs Calculated vs Depth')
            ax.set_xlabel('UW (kN/m3)')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

        # Plot the other selected lab test columns
        other_columns = [col for col in selected_lab_columns if col not in ['SUW_Measured_kN/m3', 'SUW_Calculated_kN_m3', 'UW_Measured_kN_m3', 'UW_Calculated_kN_m3']]
        
        for col in other_columns:
            if col in df_zone_lab.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                lab_data = df_zone_lab[[col, depth_col_lab]].dropna()
                lab_data = lab_data[lab_data[col] > 0]  # Exclude 0 values
                ax.scatter(lab_data[col], lab_data[depth_col_lab], label=f'{col}', marker='o')
                ax.invert_yaxis()
                ax.set_ylabel('Depth (m)')
                ax.set_title(f'{col} vs Depth')
                ax.set_xlabel(f'{col}')
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
                
st.title('Geotechnical Test Visualization and Processing Tool')

# Common file uploader for all tabs
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xlsm"])

# Sidebar for Nk low and high estimates
nk_low_sidebar = st.sidebar.number_input("Nk Low Estimate", min_value=1.0, max_value=100.0, value=15.0, key='nk_low_sidebar')
nk_high_sidebar = st.sidebar.number_input("Nk High Estimate", min_value=1.0, max_value=100.0, value=20.0, key='nk_high_sidebar')

# Sidebar for marker and plot colors
marker_color = st.sidebar.color_picker("Pick a color for markers", "#0000FF")
plot_color = st.sidebar.color_picker("Pick a color for plots", "#FF0000")

# Map reset button
if st.sidebar.button("Reset Map"):
    st.session_state['map_reset'] = True

# Add a text input in Streamlit to allow the user to specify the discretization
discretization = st.number_input("Discretization Interval (m)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

# Tabs creation
tab1, tab2, tab3, tab4 = st.tabs(["Test Visualization", "Map Display", "In-Situ Data Processing", "Data Extrapolation"])

with tab1:
    st.header('Import data from an Excel file')

    if uploaded_file is not None:
        df = load_excel_sheet(uploaded_file, 'Locations', 0)

        if df is not None:
            st.dataframe(df)

            easting_col = st.selectbox("Select the column containing Easting coordinates:", df.columns)
            northing_col = st.selectbox("Select the column containing Northing coordinates:", df.columns)
            type_col = st.selectbox("Select the column containing types:", df.columns)
            location_id_col = st.selectbox("Select the column containing Location IDs:", df.columns)

            crs = st.selectbox("Select the coordinate system:", ['utm'])
            zone = st.number_input("Enter the UTM zone:", min_value=1, max_value=60, value=21)

            if st.button("Display map"):
                st.session_state['df_location'] = df
                st.session_state['easting_col'] = easting_col
                st.session_state['northing_col'] = northing_col
                st.session_state['type_col'] = type_col
                st.session_state['location_id_col'] = location_id_col
                st.session_state['crs'] = crs
                st.session_state['zone'] = zone

with tab2:
    st.header('Map Display')

    if 'df_location' in st.session_state:
        with st.spinner('Loading map...'):
            map_data = st_folium(plot_map(st.session_state['df_location'], 
                                          st.session_state['easting_col'], 
                                          st.session_state['northing_col'], 
                                          st.session_state['type_col'], 
                                          st.session_state['location_id_col'], 
                                          st.session_state['crs'], 
                                          st.session_state['zone'],
                                          marker_color=marker_color),
                                 key="selected_rectangle")

        if map_data and 'last_active_drawing' in map_data and map_data['last_active_drawing']:
            with st.spinner('Processing drawn shapes...'):
                selected_location_ids = []
                feature = map_data['last_active_drawing']
                if feature['geometry']['type'] in ['Polygon', 'Rectangle', 'Circle']:
                    if feature['geometry']['type'] == 'Polygon':
                        polygon_coords = feature['geometry']['coordinates'][0]
                        polygon = Polygon(polygon_coords)
                    elif feature['geometry']['type'] == 'Rectangle':
                        rectangle_coords = feature['geometry']['coordinates'][0]
                        min_lon, min_lat = rectangle_coords[0]
                        max_lon, max_lat = rectangle_coords[2]
                        polygon = box(min_lon, min_lat, max_lon, max_lat)
                    elif feature['geometry']['type'] == 'Circle':
                        center_lon, center_lat = feature['geometry']['coordinates']
                        radius = feature['properties']['radius']
                        polygon = Point(center_lon, center_lat).buffer(radius)

                    # Check geotechnical points within the polygon
                    points_to_check = zip(st.session_state['df_location'][st.session_state['easting_col']], 
                                          st.session_state['df_location'][st.session_state['northing_col']])
                    for idx, (easting, northing) in enumerate(points_to_check):
                        lat, lon = convert_easting_northing_to_lat_lon(easting, northing, st.session_state['crs'], st.session_state['zone'])
                        point = Point(lon, lat)
                        if polygon.contains(point):
                            location_id = st.session_state['df_location'].iloc[idx][st.session_state['location_id_col']]
                            selected_location_ids.append(location_id)

                # Create a container for the results
                with st.container():
                    # Refresh map with new markers
                    st_folium(plot_map(st.session_state['df_location'], 
                                       st.session_state['easting_col'], 
                                       st.session_state['northing_col'], 
                                       st.session_state['type_col'], 
                                       st.session_state['location_id_col'], 
                                       st.session_state['crs'], 
                                       st.session_state['zone'],
                                       selected_location_ids,
                                       marker_color=marker_color),
                              width=700, height=500, key="map_with_updated_markers")

                    # Display the selected Location IDs
                    if selected_location_ids:
                        st.write("Selected Location IDs:")
                        result_df = st.session_state['df_location'][st.session_state['df_location'][st.session_state['location_id_col']].isin(selected_location_ids)]
                        st.dataframe(result_df)

                        # Save selected location IDs for in-situ data processing
                        st.session_state['selected_location_ids'] = selected_location_ids

                        # Automatically display Su vs Depth for selected location IDs
                        df_all_cpts = load_all_cpts_data(uploaded_file, 'CPT ID', 'Depth', 'qnet')
                        df_all_lab = load_all_lab_data(uploaded_file, 'Depth (m)', 'Location_ID', ['PP (kPa)', 'TV (kPa)', 'LV (kPa)', 'LVr (kPa)', 'LVres (kPa)', 'FC (kPa)', 'FCr (kPa)', 'HV (kPa)', 'UU (kPa)', 'UUr (kPa)', 'DSS (kPa)'])

                        if df_all_cpts is not None and df_all_lab is not None:
                            df_all_lab = check_and_clean_data(df_all_lab, ['Depth (m)', 'PP (kPa)', 'TV (kPa)', 'LV (kPa)', 'LVr (kPa)', 'LVres (kPa)', 'FC (kPa)', 'FCr (kPa)', 'HV (kPa)', 'UU (kPa)', 'UUr (kPa)', 'DSS (kPa)'])

                            # Add toggles for displaying CPTs and percentiles
                            show_cpts = st.checkbox("Show CPTs", value=True, key='show_cpts_tab2')
                            show_percentiles = st.checkbox("Show P5, P50, P95 Percentiles", value=True, key='show_percentiles_tab2')
                            
                            # Add selection for lab tests to plot
                            selected_lab_tests = st.multiselect(
                                "Select Lab Tests to Display",
                                ['PP (kPa)', 'TV (kPa)', 'LV (kPa)', 'LVr (kPa)', 'LVres (kPa)', 'FC (kPa)', 'FCr (kPa)', 'HV (kPa)', 'UU (kPa)', 'UUr (kPa)', 'DSS (kPa)'],
                                default=['PP (kPa)', 'TV (kPa)', 'LV (kPa)', 'LVr (kPa)', 'LVres (kPa)', 'FC (kPa)', 'FCr (kPa)', 'HV (kPa)', 'UU (kPa)', 'UUr (kPa)', 'DSS (kPa)'],
                                key='multiselect_tab2'
                            )

                            # Display the plots directly below the map
                            st.subheader('Su vs Depth Plots')
                            plot_su_vs_depth_by_zone(df_all_cpts, 'CPT ID', 'Depth', 'qnet', selected_location_ids, nk_low_sidebar, nk_high_sidebar, discretization, show_cpts, show_percentiles, df_lab=df_all_lab, lab_depth_col='Depth (m)', lab_columns=selected_lab_tests)

                            # Display the count of lab tests in the selected region
                            lab_tests_count = df_all_lab[df_all_lab['Location_ID'].isin(selected_location_ids)].shape[0]
                            st.write(f"Number of Lab Tests in the selected region: {lab_tests_count}")

                            # Download selected data
                            st.markdown(download_data(result_df, "selected_location_ids.csv"), unsafe_allow_html=True)
                            st.markdown(download_data(df_all_cpts[df_all_cpts['CPT ID'].isin(selected_location_ids)], "selected_cpt_data.csv"), unsafe_allow_html=True)
                            st.markdown(download_data(df_all_lab[df_all_lab['Location_ID'].isin(selected_location_ids)], "selected_lab_data.csv"), unsafe_allow_html=True)
                    else:
                        st.write("No locations found in the selected area.")
        else:
            st.write("Draw a shape on the map to check points within it.")
    else:
        st.write("Please load data in the 'Test Visualization' tab first.")

with tab3:
    st.header('In-Situ Data Processing')
    
    if uploaded_file is not None:
        insitu_tabs = st.tabs(["Su vs Depth", "Additional Lab Tests"])

        with insitu_tabs[0]:
            st.subheader('Su vs Depth')
            nk_low_insitu = st.number_input("Nk Low Estimate", min_value=1.0, max_value=100.0, value=15.0, key='nk_low_insitu')
            nk_high_insitu = st.number_input("Nk High Estimate", min_value=1.0, max_value=100.0, value=20.0, key='nk_high_insitu')
            include_lab_tests_insitu = st.checkbox("Include Lab Tests in In-Situ Data Processing", value=True, key='include_lab_tests_tab3')

            df_all_cpts = load_all_cpts_data(uploaded_file, 'CPT ID', 'Depth', 'qnet')
            df_all_lab = load_all_lab_data(uploaded_file, 'Depth (m)', 'Location_ID', ['PP (kPa)', 'TV (kPa)', 'LV (kPa)', 'LVr (kPa)', 'LVres (kPa)', 'FC (kPa)', 'FCr (kPa)', 'HV (kPa)', 'UU (kPa)', 'UUr (kPa)', 'DSS (kPa)'])

            if df_all_cpts is not None and df_all_lab is not None:
                df_all_lab = check_and_clean_data(df_all_lab, ['Depth (m)', 'PP (kPa)', 'TV (kPa)', 'LV (kPa)', 'LVr (kPa)', 'LVres (kPa)', 'FC (kPa)', 'FCr (kPa)', 'HV (kPa)', 'UU (kPa)', 'UUr (kPa)', 'DSS (kPa)'])
                selected_lab_tests = st.multiselect("Select Lab Tests to Display", ['PP (kPa)', 'TV (kPa)', 'LV (kPa)', 'LVr (kPa)', 'LVres (kPa)', 'FC (kPa)', 'FCr (kPa)', 'HV (kPa)', 'UU (kPa)', 'UUr (kPa)', 'DSS (kPa)'], default=['PP (kPa)', 'TV (kPa)', 'LV (kPa)', 'LVr (kPa)', 'LVres (kPa)', 'FC (kPa)', 'FCr (kPa)', 'HV (kPa)', 'UU (kPa)', 'UUr (kPa)', 'DSS (kPa)'], key='multiselect_tab3')
                plot_su_vs_depth_by_zone(df_all_cpts, 'CPT ID', 'Depth', 'qnet', df_all_cpts['CPT ID'].unique(), nk_low_insitu, nk_high_insitu, discretization, True, True, df_lab=df_all_lab, lab_depth_col='Depth (m)', lab_columns=selected_lab_tests)

        with insitu_tabs[1]:
            st.subheader('Additional Lab Tests')
            df_all_lab = load_all_lab_data(uploaded_file, 'Depth (m)', 'Location_ID', ['W (%)', 'SUW_Measured_kN/m3', 'SUW_Calculated_kN_m3', 'UW_Measured_kN_m3', 'UW_Calculated_kN_m3', 'IP'])

            if df_all_lab is not None:
                df_all_lab = check_and_clean_data(df_all_lab, ['Depth (m)', 'W (%)', 'SUW_Measured_kN/m3', 'SUW_Calculated_kN_m3', 'UW_Measured_kN_m3', 'UW_Calculated_kN_m3', 'IP'])
                plot_additional_lab_tests(df_all_lab, 'Depth (m)', ['W (%)', 'SUW_Measured_kN/m3', 'SUW_Calculated_kN_m3', 'UW_Measured_kN_m3', 'UW_Calculated_kN_m3', 'IP'])

