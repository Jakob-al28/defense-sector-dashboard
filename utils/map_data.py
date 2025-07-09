import pandas as pd
import json
import streamlit as st
from pandas import DataFrame
from typing import Optional

COUNTRY_NAME_MAPPING = {
    'United States': 'United States of America',
    'Czechia': 'Czech Republic', 
    'Turkiye': 'Turkey',
    'Korea, Rep.': 'South Korea',
    'Korea, Dem. People\'s Rep.': 'North Korea',
    'Russian Federation': 'Russia',
    'Iran, Islamic Rep.': 'Iran',
    'Egypt, Arab Rep.': 'Egypt',
    'Venezuela, RB': 'Venezuela',
    'Yemen, Rep.': 'Yemen',
    'Syrian Arab Republic': 'Syria',
    'Congo, Dem. Rep.': 'Democratic Republic of the Congo',
    'Congo, Rep.': 'Republic of the Congo',
    'Cote d\'Ivoire': 'Ivory Coast',
    'Cabo Verde': 'Cape Verde',
    'Bahamas, The': 'Bahamas',
    'Gambia, The': 'Gambia',
    'North Macedonia': 'Macedonia',
    'Brunei Darussalam': 'Brunei',
    'Eswatini': 'Swaziland',
    'Timor-Leste': 'East Timor',
    "Viet Nam": "Vietnam",
    'The Bahamas': 'Bahamas',
    'The Gambia': 'Gambia',
    'Czech Republic': 'Czech Republic',
    'Korea': 'South Korea',
    "Korea, South": "South Korea",
    "German Democratic Republic": "Germany",
    'Islamic Republic of Iran': 'Iran',
    'Russia': 'Russia',
    'Democratic Republic of the Congo': 'Democratic Republic of the Congo',
    'Republic of Congo': 'Republic of the Congo',
    'Côte d\'Ivoire': 'Ivory Coast',
    'Hong Kong SAR': 'Hong Kong',
    'Macao SAR': 'Macao',
    'Lao P.D.R.': 'Laos',
    'Kyrgyz Republic': 'Kyrgyzstan',
    'Slovak Republic': 'Slovakia',
    'Taiwan Province of China': 'Taiwan',
    'West Bank and Gaza': 'Palestine',
    'Syria': 'Syria',
    'Türkiye': 'Turkey'
}

@st.cache_data
def load_gdp_data() -> Optional[DataFrame]:
    """
    Load raw GDP data for 2024 from IMF World Economic Outlook data
    
    Returns:
        Optional[DataFrame]: DataFrame with columns ['Country', 'GDP', 'GDP_Formatted'] or None if error
    """
    try:        
        df = pd.read_csv("data/raw_gdp.csv")
        
        gdp_filter = (
            (df['Subject Descriptor'] == 'Gross domestic product, current prices') &
            (df['Units'] == 'U.S. dollars')
        )
        gdp_df = df[gdp_filter].copy()
        
        if gdp_df.empty:
            return None

        year_cols = [str(y) for y in range(1980, 2025)]
        keep_cols = ['Country'] + year_cols
        result_df = gdp_df.loc[:, keep_cols].copy()     
        result_df.rename(columns={year: f"{year}_gdp" for year in year_cols}, inplace=True)        
        result_df['Country'] = result_df['Country'].replace(COUNTRY_NAME_MAPPING)
        return result_df

    except Exception as e:
        return None

@st.cache_data
def load_gdp_change_data() -> Optional[DataFrame]:
    """
    Load GDP change data for a specific year
    
    Args:
        target_year (int): Year to load GDP change data for
    
    Returns:
        Optional[DataFrame]: DataFrame with GDP change data or None if error
    """

    # Read the GDP change data
    df = pd.read_csv("data/gpd_change_delta_data.csv")

    year_cols = [str(y) for y in range(2015, 2025)]
    keep_cols = ['Country Name'] + year_cols
    result_df = df.loc[:, keep_cols].copy()  
    result_df = result_df.rename(columns={'Country Name': 'Country'})  
    result_df.rename(columns={year: f"{year}_gdp" for year in year_cols}, inplace=True)      
    result_df.loc[:, 'Country'] = result_df['Country'].replace(COUNTRY_NAME_MAPPING)
    
    region_pattern = 'region|income|World|IDA|IBRD|EMU|Arab|Asia|Europe|Latin|North|Sub|Pacific|Caribbean'
    result_df = result_df[~result_df['Country'].str.contains(region_pattern, case=False, na=False)]

    return result_df

@st.cache_data
def load_peace_index_data() -> Optional[DataFrame]:
    """
    Load peace index data
    """
    df = pd.read_csv("data/peace_index.csv", 
                    encoding='utf-16', 
                    sep='\t', 
                    engine='python'  
                    ) 
    keep_cols = ['Country', "Total_peace"]
    result_df = df.loc[:, keep_cols].copy()  
    result_df.loc[:, 'Country'] = result_df['Country'].replace(COUNTRY_NAME_MAPPING)
    return result_df

@st.cache_data
def load_sus_index_data() -> Optional[DataFrame]:
    """
    Load SUS index data
    """
    df = pd.read_csv("data/owid-co2-data.csv")
    df_filtered = df[df['year'].between(2000, 2025)]

    result_df = df_filtered.pivot(
        index='Country',
        columns='year',
        values='total_ghg'
    ).reset_index()

    year_cols = [str(year) for year in range(2000, 2026)]

    result_df.rename(columns={year: f"{year}_sus" for year in year_cols}, inplace=True)

    result_df.loc[:, 'Country'] = result_df['Country'].replace(COUNTRY_NAME_MAPPING)


    return result_df

def load_gdppc_data() -> Optional[DataFrame]:
    """
    Load GDP per capita data
    
    Returns:
        Optional[DataFrame]: DataFrame with GDP per capita data or None if error
    """
    try:
        df = pd.read_csv("data/gdp_per_capita.csv")
        df.loc[:, 'Country'] = df['Country'].replace(COUNTRY_NAME_MAPPING)

        year_cols = [str(year) for year in range(1980, 2025)]

        df.rename(columns={year: f"{year}_gdppc" for year in year_cols}, inplace=True)
        return df
    except Exception as e:
        return None

@st.cache_data
def load_data() -> Optional[DataFrame]:
    """
    Load and process SIPRI military expenditure data
    
    Args:
        target_year (int): Year to load data for (defaults to 2024)
    
    Returns:
        Optional[DataFrame]: DataFrame with military spending data or None if error
    """
    try:
        df = pd.read_excel(
            'data/SIPRI-Milex-data-1949-2024_2.xlsx',
            sheet_name='Share of govt. spending cleaned'
        )
        
        region_pattern = 'region|income|World|IDA|IBRD|EMU|Arab|Asia|Europe|Latin|North|Sub|Pacific|Caribbean|Africa'
        df_clean = df[~df['Country'].str.contains(region_pattern, case=False, na=False)]
        df_clean.loc[:, 'Country'] = df_clean['Country'].replace(COUNTRY_NAME_MAPPING)

        gdp_data = load_gdp_data()

        if gdp_data is not None:
            df_clean = df_clean.merge(gdp_data, on='Country', how='left', suffixes=('', '_gdp'))
        
        # Load and merge GDP change data
        gdp_change_data = load_gdp_change_data()
        if gdp_change_data is not None:
            df_clean = df_clean.merge(gdp_change_data, on='Country', how='left', suffixes=('', '_delta'))


        sus_index_df = load_sus_index_data()
        if sus_index_df is not None:
            df_clean = df_clean.merge(sus_index_df, on='Country', how='left', suffixes=('', '_co2'))

        peace_index_df = load_peace_index_data()
        if peace_index_df is not None:
            df_clean = df_clean.merge(peace_index_df, on='Country', how='left', suffixes=('', '_peace'))
    
        gdppc = load_gdppc_data()
        if gdppc is not None:
            df_clean = df_clean.merge(gdppc, on='Country', how='left', suffixes=('', '_gdppc'))
        return df_clean

    except Exception as e:
        print(f"Error in load_military_data: {e}")
        return None

@st.cache_data
def load_geojson():
    """Load the GeoJSON file for country boundaries"""
    try:
        with open("data/World_Countries_(Generalized).geojson") as f:
            countries_geo = json.load(f)
            
        return countries_geo
    except Exception as e:
        return None