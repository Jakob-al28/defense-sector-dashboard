import streamlit as st 
import pandas as pd
import plotly.express as px
import numpy as np
import json
import os
from typing import Optional, Dict, Any, Tuple
from pandas import DataFrame, Series
from utils.map_data import load_data

def clean_non_numeric_values(df: DataFrame, value_columns: list) -> DataFrame:
    """
    Clean non-numeric values from specified columns, treating them as NaN.
    
    Args:
        df: DataFrame to clean
        value_columns: List of column names containing numeric data
    
    Returns:
        DataFrame with non-numeric values converted to NaN
    """
    df_clean = df.copy()
    
    non_numeric_patterns = ['xxx', '...', '--', 'n.a.', 'n/a', 'NA', 'N/A', '', ' ']
    
    for col in value_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)
            
            for pattern in non_numeric_patterns:
                df_clean[col] = df_clean[col].replace(pattern, np.nan)
            
            df_clean[col] = df_clean[col].str.replace(',', '', regex=False)
            
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean

def get_year_columns(df: DataFrame, suffix: str = '') -> list:
    """
    Extract year columns from dataframe based on suffix.
    
    Args:
        df: DataFrame to extract columns from
        suffix: Suffix to filter columns (e.g., '_gdp', '_co2')
    
    Returns:
        List of year column names (as they appear in the dataframe)
    """
    if suffix:
        year_cols = [col for col in df.columns if str(col).endswith(suffix) and any(str(year) in str(col) for year in range(1949, 2025))]
    else:
        year_cols = []
        for col in df.columns:
            if col == 'Country':
                continue
            # Handle both integer and string year columns
            col_str = str(col)
            if col_str.isdigit() and 1949 <= int(col_str) <= 2024:
                year_cols.append(col)  
    
    # Sort by year value, not string representation
    return sorted(year_cols, key=lambda x: int(str(x).split('_')[0]) if str(x).split('_')[0].isdigit() else 0)

def linear_regression_extrapolation(series: Series) -> Series:
    """
    Apply backward and forward linear regression extrapolation for edge gaps.
    
    Args:
        series: Time series data with potential missing values
    
    Returns:
        Series with edge gaps filled using linear trend extrapolation
    """
    if series.isna().all():
        return series
    
    result = series.copy()
    
    first_valid_idx = series.first_valid_index()
    last_valid_idx = series.last_valid_index()
    
    if first_valid_idx is None or last_valid_idx is None:
        return result
    
    series_values = series.values
    series_index = series.index.tolist()
    
    try:
        first_valid_pos = series_index.index(first_valid_idx)
        last_valid_pos = series_index.index(last_valid_idx)
    except ValueError:
        return result
    
    if first_valid_pos > 0:
        end_pos = min(first_valid_pos + 5, len(series))
        valid_data = []
        valid_positions = []
        
        for i in range(first_valid_pos, end_pos):
            if not pd.isna(series_values[i]):
                valid_data.append(series_values[i])
                valid_positions.append(i)
        
        if len(valid_data) >= 2:
            try:
                slope, intercept = np.polyfit(valid_positions, valid_data, 1)
                
                slope = slope * 0.5
                
                for i in range(first_valid_pos):
                    extrapolated_value = slope * i + intercept
                    extrapolated_value = max(extrapolated_value, 0.03)
                    result.iloc[i] = extrapolated_value
                    
            except (np.linalg.LinAlgError, ValueError):
                if len(valid_data) >= 2:
                    slope = (valid_data[1] - valid_data[0]) / (valid_positions[1] - valid_positions[0])
                    slope = slope * 0.5
                    for i in range(first_valid_pos):
                        steps_back = first_valid_pos - i
                        extrapolated_value = valid_data[0] - slope * steps_back
                        extrapolated_value = max(extrapolated_value, 0.03)
                        result.iloc[i] = extrapolated_value
    
    if last_valid_pos < len(series) - 1:
        start_pos = max(0, last_valid_pos - 4)
        valid_data = []
        valid_positions = []
        
        for i in range(start_pos, last_valid_pos + 1):
            if not pd.isna(series_values[i]):
                valid_data.append(series_values[i])
                valid_positions.append(i)
        
        if len(valid_data) >= 2:
            try:
                slope, intercept = np.polyfit(valid_positions, valid_data, 1)
                
                slope = slope * 0.5
                
                for i in range(last_valid_pos + 1, len(series)):
                    extrapolated_value = slope * i + intercept
                    extrapolated_value = max(extrapolated_value, 0.03)
                    result.iloc[i] = extrapolated_value
            except (np.linalg.LinAlgError, ValueError):
                fill_value = result.iloc[last_valid_pos]
                for i in range(last_valid_pos + 1, len(series)):
                    result.iloc[i] = fill_value
        
        if start_pos > 0:
            steps_back = 1
            for i in range(start_pos - 1, -1, -1):
                if start_pos + 1 < len(series):
                    try:
                        slope = (result.iloc[start_pos + 1] - result.iloc[start_pos])
                        
                        slope = slope * 0.5
                        
                        extrapolated_value = result.iloc[start_pos] - slope * steps_back
                        extrapolated_value = max(extrapolated_value, 0.03)
                        result.iloc[i] = extrapolated_value
                        steps_back += 1
                    except:
                        result.iloc[i] = result.iloc[start_pos]
                else:
                    result.iloc[i] = result.iloc[start_pos]
        
        if last_valid_pos < len(series) - 1:
            steps_forward = 1
            for i in range(last_valid_pos + 1, len(series)):
                if last_valid_pos > 0:
                    try:
                        slope = (result.iloc[last_valid_pos] - result.iloc[last_valid_pos - 1])
                        
                        slope = slope * 0.5
                        
                        extrapolated_value = result.iloc[last_valid_pos] + slope * steps_forward
                        extrapolated_value = max(extrapolated_value, 0.03)
                        result.iloc[i] = extrapolated_value
                        steps_forward += 1
                    except:
                        result.iloc[i] = result.iloc[last_valid_pos]
                else:
                    result.iloc[i] = result.iloc[last_valid_pos]
    
    return result

def interpolate_time_series(series: Series, method: str = 'auto') -> Tuple[Series, Series]:
    """
    Interpolate missing values in a time series using statistical methods.
    
    Args:
        series: Time series data with potential missing values
        method: Interpolation method ('linear', 'polynomial', 'spline', 'auto')
    
    Returns:
        Tuple of (interpolated_series, extrapolation_markers)
    """
    if series.isna().all():
        return series, pd.Series([False] * len(series), index=series.index)
    
    extrapolation_markers = series.isna().copy()
    
    valid_count = series.count()
    
    if valid_count == 0:
        return series, extrapolation_markers
    
    if valid_count == 1:
        interpolated = series.bfill().ffill()
        return interpolated, extrapolation_markers
    
    interpolated = series
    if method == 'auto':
        if valid_count <= 3:
            method = 'linear'
        elif valid_count <= 10:
            method = 'polynomial'
        else:
            method = 'spline'
        
    
    interpolated = interpolated.copy()
    
    if interpolated.dtype == 'object':
        interpolated = interpolated.infer_objects(copy=False)

    try:
        if method == 'linear':
            interpolated = interpolated.interpolate(method='linear')
            
        elif method == 'polynomial':
            degree = min(3, valid_count - 1)
            if degree > 0:
                interpolated = interpolated.interpolate(method='polynomial', order=degree)
            else:
                interpolated = interpolated.interpolate(method='linear')
                
        elif method == 'spline':
            if valid_count >= 4:
                interpolated = interpolated.interpolate(method='spline', order=3)
            else:
                interpolated = interpolated.interpolate(method='linear')
    
    except Exception:
        interpolated = interpolated.interpolate(method='linear')
    
    if interpolated.isna().any():
        interpolated = interpolated.bfill()
        interpolated = interpolated.ffill()
    
    return interpolated, extrapolation_markers

def extrapolate_dataframe_time_series(df: DataFrame, data_type: str) -> Tuple[DataFrame, DataFrame]:
    """
    Extrapolate missing values for all countries in a dataframe time series.
    
    Args:
        df: DataFrame with countries as rows and years as columns
        data_type: Type of data for appropriate processing ('military', 'gdp', 'emissions', etc.)
    
    Returns:
        Tuple of (extrapolated_dataframe, extrapolation_markers_dataframe)
    """
    if data_type == 'military':
        year_cols = get_year_columns(df, '')
    elif data_type == 'gdp':
        year_cols = get_year_columns(df, '_gdp')
    elif data_type == 'gdp_change':
        year_cols = get_year_columns(df, '_delta')
    elif data_type == 'emissions':
        year_cols = get_year_columns(df, '_co2')
    elif data_type == 'gdppc':
        year_cols = get_year_columns(df, '_gdppc')
    else:
        year_cols = [col for col in df.columns if col != 'Country']
    
    if not year_cols:
        return df, pd.DataFrame()
    
    df_clean = clean_non_numeric_values(df, year_cols)

    df_extrapolated = df_clean.copy()
    df_markers = pd.DataFrame(False, index=df_clean.index, columns=pd.Index(year_cols))
    
    for idx, row in df_clean.iterrows():
        country_data = pd.Series(row[year_cols], index=year_cols)
        
        if country_data.isna().all():
            continue
            
        if data_type in ['military', 'gdp', 'gdppc']:
            method = 'auto'  
        elif data_type == 'emissions':
            method = 'spline' 
        else:
            method = 'linear' 
        
        interpolated_data, extrapolation_markers = interpolate_time_series(country_data, method)
        
        df_extrapolated.loc[idx, year_cols] = interpolated_data.values
        df_markers.loc[idx, year_cols] = extrapolation_markers.values
    
    return df_extrapolated, df_markers

def render_time_series(data_type, clicked_countries=None):
    """
    Render time series data with extrapolation for missing values.
    
    Args:
        clicked_countries: List of pre-selected countries
    
    Returns:
        Dictionary containing processed dataframes and extrapolation markers
    """
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please check your data files.")
        return None
    
    df = df.drop(114, errors='ignore')
    
    df_military = df.iloc[:, :38]
    df_gdp = df[['Country']].join(df.iloc[:, 38:83])
    df_gdp_change = df[['Country']].join(df.iloc[:, 83:93])
    df_emissions = df[['Country']].join(df.iloc[:, 93:117])
    df_peace = df[['Country', df.columns[117]]]    
    df_gdppc = df[['Country']].join(df.iloc[:, 117:])
    
    markers = None
    if data_type == 'Military Expenditure':
        df, markers = extrapolate_dataframe_time_series(df_military, 'military')
        st.session_state.tab3_df = df
        st.session_state.tab3_markers = markers
    elif data_type == 'GDP Total':
        df, markers = extrapolate_dataframe_time_series(df_gdp, 'gdp')
    elif data_type == 'GDP Change From Last Year':
        df, markers = extrapolate_dataframe_time_series(df_gdp_change, 'gdp_change')
    elif data_type == 'CO2 Emissions':
        df, markers = extrapolate_dataframe_time_series(df_emissions, 'emissions')
    elif data_type == 'GDP per capita':
        df, markers = extrapolate_dataframe_time_series(df_gdppc, 'gdppc')
    elif data_type == 'Peace Index':
        return
    else:
        df, markers = extrapolate_dataframe_time_series(df_military, 'military')



        data_type = 'Military Expenditure'
    # Get all available countries and sort alphabetically
    all_countries = sorted(df['Country'].unique().tolist())
    
    # Initialize session state for selected countries if not exists
    if 'selected_countries_timeseries' not in st.session_state:
        st.session_state.selected_countries_timeseries = []
    
    if 'time_series_flag' not in st.session_state:
        st.session_state.time_series_flag = True

    if st.session_state.time_series_flag:
        st.session_state.selected_countries_timeseries.extend(["Germany", "Poland", "United Kingdom", "France", "United States"])
        st.session_state.time_series_flag = False


    if clicked_countries:
        countries_added = False
        for country in clicked_countries:
            if country not in st.session_state.selected_countries_timeseries:
                st.session_state.selected_countries_timeseries.append(country)
                countries_added = True
        
        if countries_added:
            st.rerun()

    st.markdown("#### Select Countries for Time Series Comparison or click on the map")  

    with st.expander("Country Selection", expanded=False):
        # Get countries organized by continent
        continent_countries = get_countries_by_continent()
        
        # Find unmapped countries (countries in data but not in continent mapping)
        all_mapped_countries = []
        for countries in continent_countries.values():
            all_mapped_countries.extend(countries)
        
        unmapped_countries = [country for country in all_countries if country not in all_mapped_countries]
        
        # Add "Other" tab if there are unmapped countries
        tab_names = list(continent_countries.keys())
        
        continent_tabs = st.tabs(tab_names)
        
        # Display continent tabs
        for i, (continent, countries) in enumerate(continent_countries.items()):
            with continent_tabs[i]:
                # Filter to only show countries that exist in our data
                available_countries = [country for country in countries if country in all_countries]
                
                if not available_countries:
                    st.write(f"No data available for {continent}")
                    continue
                
                # Display countries in 4 columns
                num_cols = 4
                cols = st.columns(num_cols)
                
                for j, country in enumerate(available_countries):
                    col_idx = j % num_cols
                    with cols[col_idx]:
                        is_selected = country in st.session_state.selected_countries_timeseries
                        
                        checkbox_key = f"checkbox_{country}_{continent}"
                        checked = st.checkbox(
                            country,
                            value=is_selected,
                            key=checkbox_key
                        )
                        
                        if checked and country not in st.session_state.selected_countries_timeseries:
                            st.session_state.selected_countries_timeseries.append(country)
                        elif not checked and country in st.session_state.selected_countries_timeseries:
                            st.session_state.selected_countries_timeseries.remove(country)
        
    st.info(f"Missing values were estimated using interpolation methods (linear, polynomial, Trend Analysis). In the time series below, extrapolated data points are shown as orange diamonds while original data appears as blue circles.")
    active_countries = st.session_state.selected_countries_timeseries

    create_time_series_plot(df, markers, active_countries, data_type, title=data_type)
 

def create_time_series_plot(data_df: DataFrame, markers_df: DataFrame, countries: Optional[list], data_type: str = 'Military Expenditure', title: Optional[str] = None):
    """
    Create and display a time series plot with multiple countries on the same plot.
    
    Args:
        data_df: DataFrame with extrapolated values
        markers_df: DataFrame with extrapolation markers
        countries: List of country names to plot
        data_type: Type of data to plot
        title: Custom title for the plot
    """
    fig = px.line()
    if data_type:
        title = "Time Series of " + data_type

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    data_multiplier = 1
    hover_suffix = ""
    
    if data_type == 'Military Expenditure' or data_type == 'GDP Change From Last Year':
        data_multiplier = 100
        hover_suffix = " %"
    elif data_type == 'GDP Total':
        hover_suffix = " B"
    elif data_type == 'CO2 Emissions':
        hover_suffix = " B. T."
    elif data_type == 'GDP per capita':
        hover_suffix = " USD"


    if countries:
        for i, country in enumerate(countries):
            if country not in data_df['Country'].values:
                continue
            
            country_color = colors[i % len(colors)]
            
            country_idx = data_df[data_df['Country'] == country].index[0]
            
            year_cols = [col for col in data_df.columns if col != 'Country']
            
            years = []
            values = []
            markers = []
            
            for col in year_cols:
                year_str = col
                if type(col) != int:
                    year_str = col.split('_')[0]
                try:
                    year = int(year_str)
                    years.append(year)
                    raw_value = data_df.loc[country_idx, col]
                    transformed_value = raw_value * data_multiplier
                    values.append(transformed_value)
                    if col in markers_df.columns:
                        markers.append(markers_df.loc[country_idx, col])
                    else:
                        markers.append(False)
                except ValueError:
                    continue
            
            if not years:
                continue
            
            original_years = [year for year, is_extrapolated in zip(years, markers) if not is_extrapolated]
            original_values = [val for val, is_extrapolated in zip(values, markers) if not is_extrapolated]
            
            extrapolated_years = [year for year, is_extrapolated in zip(years, markers) if is_extrapolated]
            extrapolated_values = [val for val, is_extrapolated in zip(values, markers) if is_extrapolated]
        
            fig.add_scatter(
                x=years,
                y=values,
                mode='lines',
                name=f'{country}',
                line=dict(width=2, color=country_color),
                showlegend=False,
                hoverinfo='skip'
            )
            
            # Set appropriate decimal places based on data type
            if data_type == 'Military Expenditure' or data_type == 'GDP Change From Last Year':
                value_format = '%{y:,.1f}'  # 1 decimal place for percentages
            else:
                value_format = '%{y:,.0f}'  # 0 decimal places for other data
            
            if original_years:
                fig.add_scatter(
                    x=original_years,
                    y=original_values,
                    mode='markers',
                    name=f'{country}',
                    line=dict(width=0), 
                    marker=dict(size=6, color=country_color),
                    showlegend=True,
                    hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Value: ' + value_format + hover_suffix + '<extra></extra>'
                )
            
            if extrapolated_years:
                fig.add_scatter(
                    x=extrapolated_years,
                    y=extrapolated_values,
                    mode='markers',
                    name=f'{country} (Extrapolated)',
                    line=dict(width=0), 
                    marker=dict(symbol='diamond', size=6, color='white'),
                    showlegend=False,
                    hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Value: ' + value_format + hover_suffix + '<extra></extra>'
                )
        
        y_axis_title = f'{data_type} Value'
        if hover_suffix:
            y_axis_title += f' ({hover_suffix})'
        
        fig.update_layout(
            title=title or f'{data_type} Data Comparison',
            xaxis_title='Year',
            yaxis_title=y_axis_title,
            hovermode='x unified',
            showlegend=True,
            template='plotly_white',
            title_font_size=18,
            font=dict(size=16),
            xaxis=dict(title_font_size=16),
            yaxis=dict(title_font_size=16, tickformat=',.0f'),
            legend=dict(font=dict(size=16)),  
            hoverlabel=dict(font_size=16) 
        )
        
        st.plotly_chart(fig, use_container_width=True)

def save_country_preset(countries: list, preset_name: Optional[str] = None) -> bool:
    """
    Save a country selection as a preset to a JSON file.
    
    Args:
        countries: List of country names
        preset_name: Custom name for the preset (auto-generated if None)
    
    Returns:
        True if saved successfully, False otherwise
    """
    if not countries:
        return False
        
    if not preset_name:
        if len(countries) <= 3:
            preset_name = "_".join(countries)
        else:
            preset_name = f"{len(countries)}_Countries_{countries[0]}_etc"
        
        if len(preset_name) > 50:
            preset_name = preset_name[:47] + "..."
    
    preset_file = "country_presets.json"
    
    try:
        if os.path.exists(preset_file):
            with open(preset_file, 'r', encoding='utf-8') as f:
                presets = json.load(f)
        else:
            presets = {}
        
        presets[preset_name] = {
            "countries": countries,
            "created_at": pd.Timestamp.now().isoformat()
        }
        
        with open(preset_file, 'w', encoding='utf-8') as f:
            json.dump(presets, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        st.error(f"Failed to save preset: {e}")
        return False

def load_country_presets() -> dict:
    """
    Load country presets from JSON file.
    
    Returns:
        Dictionary of presets {name: {countries: list, created_at: str}}
    """
    preset_file = "country_presets.json"
    
    try:
        if os.path.exists(preset_file):
            with open(preset_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        st.error(f"Failed to load presets: {e}")
        return {}

def delete_country_preset(preset_name: str) -> bool:
    """
    Delete a country preset.
    
    Args:
        preset_name: Name of the preset to delete
    
    Returns:
        True if deleted successfully, False otherwise
    """
    preset_file = "country_presets.json"
    
    try:
        if os.path.exists(preset_file):
            with open(preset_file, 'r', encoding='utf-8') as f:
                presets = json.load(f)
            
            if preset_name in presets:
                del presets[preset_name]
                
                with open(preset_file, 'w', encoding='utf-8') as f:
                    json.dump(presets, f, indent=2, ensure_ascii=False)
                
                return True
        return False
        
    except Exception as e:
        st.error(f"Failed to delete preset: {e}")
        return False

def get_extrapolation_summary(data_dict: Dict) -> Dict[str, Any]:
    """
    Generate a summary of extrapolation statistics for all data types.
    
    Args:
        data_dict: Dictionary returned from render_time_series()
    
    Returns:
        Dictionary with extrapolation statistics
    """
    summary = {}
    
    for data_type in ['military', 'gdp', 'gdp_change', 'emissions', 'gdppc']:
        if data_type in data_dict and 'markers' in data_dict[data_type]:
            markers_df = data_dict[data_type]['markers']
            
            if not markers_df.empty:
                total_values = markers_df.size
                extrapolated_values = markers_df.sum().sum()
                percentage = (extrapolated_values / total_values * 100) if total_values > 0 else 0
                
                summary[data_type] = {
                    'total_values': total_values,
                    'extrapolated_values': extrapolated_values,
                    'percentage': percentage,
                    'countries_with_extrapolation': (markers_df.sum(axis=1) > 0).sum()
                }
    
    return summary

def get_countries_by_continent():
    """
    Map countries to continents with comprehensive name variations.
    
    Returns:
        Dict mapping continent names to sorted lists of countries
    """
    continent_mapping = {
        'Europe': [
            'Albania', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 
            'Croatia', 'Czech Republic', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 
            'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 
            'Luxembourg', 'Malta', 'Montenegro', 'Netherlands', 'North Macedonia', 'Macedonia',
            'Norway', 'Poland', 'Portugal', 'Romania', 'Serbia', 'Slovakia', 'Slovenia', 
            'Spain', 'Sweden', 'Switzerland', 'Ukraine', 'United Kingdom', 'UK', 'Britain',
            'Great Britain', 'Moldova', 'Russia', 'Russian Federation', 'Cyprus'
        ],
        'North America': [
            'Canada', 'United States', 'United States of America', 'USA', 'US', 'Mexico',
            'Guatemala', 'Belize', 'El Salvador', 'Honduras', 'Nicaragua', 'Costa Rica', 
            'Panama', 'Cuba', 'Jamaica', 'Haiti', 'Dominican Republic', 'Trinidad and Tobago', 
            'Barbados', 'Bahamas', 'Grenada', 'Saint Lucia', 'Saint Vincent and the Grenadines',
            'Antigua and Barbuda', 'Dominica', 'Saint Kitts and Nevis'
        ],
        'South America': [
            'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'French Guiana', 
            'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela'
        ],
        'Asia': [
            'Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 
            'Cambodia', 'China', "China, People's Republic of", 'Georgia', 'India', 'Indonesia', 
            'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 
            'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Burma', 'Nepal', 
            'North Korea', 'DPRK', 'Oman', 'Pakistan', 'Palestine', 'Philippines', 'Qatar', 
            'Saudi Arabia', 'Singapore', 'South Korea', 'Republic of Korea', 'Korea, South',
            'Sri Lanka', 'Syria', 'Taiwan', 'Tajikistan', 'Thailand', 'Timor-Leste', 'East Timor',
            'Turkey', 'Turkmenistan', 'United Arab Emirates', 'UAE', 'Uzbekistan', 'Vietnam', 'Yemen'
        ],
        'Africa': [
            'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 
            'Cape Verde', 'Central African Republic', 'Chad', 'Comoros', 'Congo, DR', 
            'Congo, Republic', 'Djibouti', 
            'Egypt', 'Equatorial Guinea', 'Eritrea', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 
            'Guinea', 'Guinea-Bissau', 'Ivory Coast', "Côte d'Ivoire", 'Kenya', 'Lesotho', 
            'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 
            'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 
            'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 
            'South Africa', 'South Sudan', 'Sudan', 'Eswatini', 'Swaziland', 'Tanzania', 
            'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'
        ],
        'Oceania': [
            'Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 
            'New Zealand', 'Palau', 'Papua New Guinea', 'Samoa', 'Solomon Islands', 
            'Tonga', 'Tuvalu', 'Vanuatu'
        ]
    }
    
    # Sort countries alphabetically within each continent and remove duplicates
    for continent in continent_mapping:
        continent_mapping[continent] = sorted(list(set(continent_mapping[continent])))
    
    return continent_mapping
