import streamlit as st
import pandas as pd
import folium
from pyproj import Proj, Transformer
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import numpy as np

@st.cache_data
def convert_easting_northing_to_lat_lon(easting, northing, crs, zone):
    proj_utm = Proj(proj=crs, zone=zone, ellps='WGS84')
    proj_latlon = Proj(proj='latlong', datum='WGS84')
    transformer = Transformer.from_proj(proj_utm, proj_latlon)
    lon, lat = transformer.transform(easting, northing)
    return lat, lon

@st.cache_data
def load_excel_sheet(uploaded_file, sheet_name, header_row):
    try:
        xls = pd.ExcelFile(uploaded_file, engine='openpyxl')
        df = pd.read_excel(xls, sheet_name=sheet_name, engine='openpyxl', header=header_row)
        return df
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier Excel : {e}")
        return None

def upload_excel_and_display(key=None):
    uploaded_file = st.file_uploader("Choisissez un fichier Excel", type=["xlsx", "xlsm"], key=key)
    if uploaded_file is not None:
        xls = pd.ExcelFile(uploaded_file, engine='openpyxl')
        sheet_names = xls.sheet_names
        selected_sheet = st.selectbox("Sélectionnez une feuille", sheet_names)
        df = load_excel_sheet(uploaded_file, selected_sheet, 2)
        if df is not None:
            st.write(f"Le fichier Excel a été chargé avec succès. Feuille {selected_sheet} sélectionnée.")
            return df
    return None

def plot_map(data, easting_col, northing_col, type_col, crs, zone):
    m = folium.Map(location=[0, 0], zoom_start=2)
    data.apply(lambda row: add_marker_to_map(m, row, easting_col, northing_col, type_col, crs, zone), axis=1)
    folium_static(m)

def add_marker_to_map(m, row, easting_col, northing_col, type_col, crs, zone):
    try:
        lat, lon = convert_easting_northing_to_lat_lon(row[easting_col], row[northing_col], crs, zone)
        if row[type_col] == 'BH':
            folium.CircleMarker(location=[lat, lon], radius=5, color='red', fill=True, fill_color='red', popup=row[type_col]).add_to(m)
        elif row[type_col] == 'BC':
            folium.RegularPolygonMarker(location=[lat, lon], number_of_sides=6, radius=6, color='orange', fill=True, fill_color='orange', popup=row[type_col]).add_to(m)
        elif row[type_col] == 'PC':
            folium.RegularPolygonMarker(location=[lat, lon], number_of_sides=4, radius=6, color='green', fill=True, fill_color='green', rotation=45, popup=row[type_col]).add_to(m)
        elif row[type_col] == 'CPT':
            folium.RegularPolygonMarker(location=[lat, lon], number_of_sides=4, radius=6, color='yellow', fill=True, fill_color='yellow', popup=row[type_col]).add_to(m)
        elif row[type_col] == 'PCPT':
            folium.RegularPolygonMarker(location=[lat, lon], number_of_sides=4, radius=6, color='white', fill=True, fill_color='white', popup=row[type_col]).add_to(m)
        elif row[type_col] == 'VC':
            folium.RegularPolygonMarker(location=[lat, lon], number_of_sides=6, radius=6, color='purple', fill=True, fill_color='purple', popup=row[type_col]).add_to(m)
        else:
            folium.Marker(location=[lat, lon], popup=row[type_col]).add_to(m)
    except Exception as e:
        st.error(f"Erreur lors de la conversion des coordonnées pour la ligne {row.name}: {e}")

def make_column_names_unique(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

@st.cache_data
def load_all_cpts_data(file, cpt_id_col, depth_col, qnet_col, zone_col):
    df = load_excel_sheet(file, 'All_CPTs', None)
    if df is not None:
        df = df.drop(index=list(range(0, 24))).reset_index(drop=True)
        df.columns = df.iloc[0]
        df = df.drop(index=[0, 1, 2]).reset_index(drop=True)
        df = make_column_names_unique(df)
        st.write("Noms des colonnes dans la feuille All_CPTs:")
        st.write(df.columns)
        selected_cols = [cpt_id_col, depth_col, qnet_col, zone_col]
        df = df[selected_cols]
        df[qnet_col] = df[qnet_col].apply(lambda x: x * 1000 if x < 10 else x)
    return df

@st.cache_data
def load_all_lab_data(file, depth_col, zone_col, columns):
    df = load_excel_sheet(file, 'All_Lab', 4)
    if df is not None:
        st.write("Noms des colonnes dans la feuille All_Lab:")
        st.write(df.columns)
        df = make_column_names_unique(df)
        selected_cols = [depth_col, zone_col] + columns
        df = df[selected_cols]
    return df

def check_and_clean_data(df, columns):
    for col in columns:
        df[col] = df[col].replace('None', 0)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def plot_su_vs_depth(df, df_lab, cpt_id_col, depth_col_cpt, depth_col_lab, qnet_col, zone_col, nk_low, nk_high, selected_cpt_ids, selected_lab_columns):
    df = df.dropna(subset=[depth_col_cpt, qnet_col])
    df[depth_col_cpt] = df[depth_col_cpt].astype(float)
    df[qnet_col] = df[qnet_col].astype(float)
    df_lab[depth_col_lab] = df_lab[depth_col_lab].astype(float)

    if df[qnet_col].max() < 100:
        df[qnet_col] = df[qnet_col] * 1000

    df['Su'] = df[qnet_col] / 10
    df['Su_low'] = df[qnet_col] / nk_low
    df['Su_high'] = df[qnet_col] / nk_high

    df = df[(df['Su'] > 0) & (df['Su_low'] > 0) & (df['Su_high'] > 0)]
    df_lab = df_lab[df_lab[selected_lab_columns].gt(0).any(axis=1)]

    zones = df[zone_col].unique()

    for zone in zones:
        fig, ax = plt.subplots(figsize=(15, 20))
        ax2 = ax.twiny()

        df_zone = df[df[zone_col] == zone]
        df_lab_zone = df_lab[df_lab[zone_col] == zone]

        for cpt_id in df_zone[cpt_id_col].unique():
            if cpt_id in selected_cpt_ids:
                subset = df_zone[df_zone[cpt_id_col] == cpt_id]
                ax.plot(subset['Su'], subset[depth_col_cpt], label=f'{cpt_id_col}: {cpt_id}')
                ax.plot(subset['Su_low'], subset[depth_col_cpt], '--', label=f'{cpt_id_col} Low Est: {cpt_id}')
                ax.plot(subset['Su_high'], subset[depth_col_cpt], ':', label=f'{cpt_id_col} High Est: {cpt_id}')

        markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'h', 'H', 'd', '+']
        for i, col in enumerate(selected_lab_columns):
            if col in df_lab_zone.columns and not df_lab_zone[col].dropna().empty:
                lab_data = df_lab_zone[[col, depth_col_lab]].dropna()
                lab_data = lab_data[lab_data[col] > 0]
                ax.scatter(lab_data[col], lab_data[depth_col_lab], label=f'{col}', marker=markers[i % len(markers)], s=100)

        ax.invert_yaxis()
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'Zone: {zone}')
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xlabel('Su (kPa)')
        if selected_cpt_ids or len(selected_lab_columns) > 0:
            ax.legend()
        ax.grid(True)
        st.pyplot(fig)

st.title('Outil de Visualisation et de Traitement des Essais Géotechniques')

with st.expander("Visualisation des essais"):

    st.header('Importer des données à partir d\'un fichier Excel')
    df = upload_excel_and_display(key='point_uploader')

    if df is not None:
        st.dataframe(df)

        easting_col = st.selectbox("Sélectionnez la colonne contenant les coordonnées Easting :", df.columns)
        northing_col = st.selectbox("Sélectionnez la colonne contenant les coordonnées Northing :", df.columns)
        type_col = st.selectbox("Sélectionnez la colonne contenant les types :", df.columns)

        selected_columns = [easting_col, northing_col, type_col]
        selected_df = df[selected_columns]

        st.write("Vous avez sélectionné les colonnes suivantes pour l'affichage :")
        st.dataframe(selected_df)

        crs = st.selectbox("Sélectionnez le système de coordonnées :", ['utm'])
        zone = st.number_input("Entrez la zone UTM :", min_value=1, max_value=60, value=31)

        col1, col2 = st.columns([4, 1])

        with col1:
            plot_map(selected_df, easting_col, northing_col, type_col, crs, zone)

        with col2:
            st.write("### Légende")
            st.markdown("""
                <div style="font-size: 14px;">
                <svg height="20" width="20">
                    <circle cx="10" cy="10" r="5" stroke="black" stroke-width="1" fill="red" />
                </svg> BH<br>
                <svg height="20" width="20">
                    <polygon points="10,2 15,8 13,15 7,15 5,8" stroke="black" stroke-width="1" fill="orange"/>
                </svg> BC<br>
                <svg height="20" width="20">
                    <polygon points="10,0 15,10 10,20 5,10" stroke="black" stroke-width="1" fill="green"/>
                </svg> PC<br>
                <svg height="20" width="20">
                    <rect x="5" y="5" width="10" height="10" stroke="black" stroke-width="1" fill="yellow"/>
                </svg> CPT<br>
                <svg height="20" width="20">
                    <rect x="5" y="5" width="10" height="10" stroke="black" stroke-width="1" fill="white"/>
                </svg> PCPT<br>
                <svg height="20" width="20">
                    <polygon points="10,2 16,7 16,13 10,18 4,13 4,7" stroke="black" stroke-width="1" fill="purple"/>
                </svg> VC<br>
                </div>
            """, unsafe_allow_html=True)

with st.expander("Traitement des données in situ"):
    st.header('Traitement des données in situ')

    uploaded_file = st.file_uploader("Choisissez un fichier Excel", type=["xlsx", "xlsm"], key='cpt_uploader')

    if uploaded_file is not None:
        cpt_id_col = st.text_input("Entrez le nom de la colonne pour 'CPT ID' :", value='CPT ID')
        depth_col_cpt = st.text_input("Entrez le nom de la colonne pour 'Depth' dans All_CPTs:", value='Depth')
        depth_col_lab = st.text_input("Entrez le nom de la colonne pour 'Depth (m)' dans All_Lab:", value='Depth (m)')
        qnet_col = st.text_input("Entrez le nom de la colonne pour 'qnet' :", value='qnet')
        zone_col = st.text_input("Entrez le nom de la colonne pour 'Zone' :", value='Zone')

        nk_low = st.number_input("Entrez Nk pour Low Estimate :", min_value=1.0, max_value=100.0, value=15.0)
        nk_high = st.number_input("Entrez Nk pour High Estimate :", min_value=1.0, max_value=100.0, value=20.0)

        df_all_cpts = load_all_cpts_data(uploaded_file, cpt_id_col, depth_col_cpt, qnet_col, zone_col)
        if df_all_cpts is not None:
            st.dataframe(df_all_cpts)

            selected_cpt_ids = st.sidebar.multiselect("Sélectionnez les CPT ID à afficher", df_all_cpts[cpt_id_col].unique())

            if st.sidebar.button("Tout afficher CPT ID"):
                selected_cpt_ids = df_all_cpts[cpt_id_col].unique().tolist()
            if st.sidebar.button("Tout décocher CPT ID"):
                selected_cpt_ids = []

            lab_columns = ['PP (kPa)', 'TV (kPa)', 'LV (kPa)', 'LVr (kPa)', 'LVres (kPa)', 'FC (kPa)', 'FCr (kPa)', 'HV (kPa)', 'UU (kPa)', 'UUr (kPa)', 'DSS (kPa)']
            df_all_lab = load_all_lab_data(uploaded_file, depth_col_lab, zone_col, lab_columns)
            if df_all_lab is not None:
                df_all_lab = check_and_clean_data(df_all_lab, ['Depth (m)', 'PP (kPa)', 'TV (kPa)', 'LV (kPa)', 'LVr (kPa)', 'LVres (kPa)', 'FC (kPa)', 'FCr (kPa)', 'HV (kPa)', 'UU (kPa)', 'UUr (kPa)', 'DSS (kPa)'])

                st.dataframe(df_all_lab)

                selected_lab_columns = st.sidebar.multiselect("Sélectionnez les essais de laboratoire à afficher", lab_columns, default=lab_columns)

                if st.sidebar.button("Tout afficher essais labo"):
                    selected_lab_columns = lab_columns
                if st.sidebar.button("Tout décocher essais labo"):
                    selected_lab_columns = []

                plot_su_vs_depth(df_all_cpts, df_all_lab, cpt_id_col, depth_col_cpt, depth_col_lab, qnet_col, zone_col, nk_low, nk_high, selected_cpt_ids, selected_lab_columns)
