import streamlit as st 
import pandas as pd
import plotly.express as px
from utils.map_data import load_data


def render_racing_bar_chart(data_type):
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
    
    if data_type == 'Military Expenditure':
        selected_df = df_military
        title = "Military Expenditure Over Time (% of GDP)"
        color_scale = "Reds"
        use_percentage = True
        beginning_year = 1995
    elif data_type == 'GDP Total':
        selected_df = df_gdp
        title = "GDP Total Over Time (Billion USD)"
        color_scale = "Blues"
        use_percentage = False
        beginning_year = 1980
    elif data_type == 'GDP Change From Last Year':
        selected_df = df_gdp_change
        title = "GDP Change From Last Year (%)"
        color_scale = "Blues"
        use_percentage = False  
        beginning_year = 2015
    elif data_type == 'CO2 Emissions':
        selected_df = df_emissions
        title = "CO2 Emissions Over Time (Million tonnes)"
        color_scale = "Oranges"
        use_percentage = False
        beginning_year = 2000
    elif data_type == 'GDP per capita':
        selected_df = df_gdppc
        title = "GDP per capita Over Time (USD)"
        color_scale = "Purples"
        use_percentage = False
        beginning_year = 1960
    elif data_type == 'Peace Index':
        st.info("Racing bar chart not available for Peace Index (single year data)")
        return
    else:
        st.error("Unknown data type")
        return
    
    data_for_chart = prepare_racing_data(selected_df, data_type, use_percentage)
    
    if data_for_chart.empty:
        st.error("No data available for racing bar chart")
        return
    
    
    create_racing_bar_chart(data_for_chart, title, color_scale, use_percentage, beginning_year, data_type)
    if data_type == 'Military Expenditure':
        st.write(
            "The eastward trend is evident here as well, and it’s interesting to note that China’s military expenditure has declined over the years."
        )
    elif data_type == 'GDP Total':
        st.write(
            "What’s striking here is China’s rapid ascent beginning in the mid-2000s, driven by the country’s economic growth." 
        )
    elif data_type == 'GDP Change From Last Year':
        pass
    elif data_type == 'CO2 Emissions':
        st.write("China quickly surpassed the United States and is now the world's largest emitter of Carbon Dioxide.")
    elif data_type == 'GDP per capita':
        pass
    else:
        st.write("unknown data type")



def prepare_racing_data(df, data_type, use_percentage=False):
    """Prepare data for racing bar chart format"""
    
    if data_type == 'Military Expenditure':
        year_columns = [col for col in df.columns if col != 'Country' and str(col).isdigit() and int(str(col)) >= 1995]
        beginning_year = 1995
    elif data_type == 'GDP Total':
        year_columns = [col for col in df.columns if col != 'Country' and str(col).endswith('_gdp') and int(str(col).split('_')[0]) >= 1980]
        beginning_year = 1980
    elif data_type == 'GDP Change From Last Year':
        year_columns = [col for col in df.columns if col != 'Country' and str(col).endswith('_gdp_delta') and int(str(col).split('_')[0]) >= 2015]
        beginning_year = 2015
    elif data_type == 'CO2 Emissions':
        year_columns = [col for col in df.columns if col != 'Country' and str(col).endswith('_co2') and int(str(col).split('_')[0]) >= 2000]
        beginning_year = 2000
    elif data_type == 'GDP per capita':
        year_columns = [col for col in df.columns if col != 'Country' and str(col).endswith('_gdppc') and int(str(col).split('_')[0]) >= 1960]
        beginning_year = 1960
    else:
        year_columns = [col for col in df.columns if col != 'Country' and str(col).isdigit() and int(str(col)) >= 1995]
        beginning_year = 1995

    if not year_columns:
        st.write("No year columns found for", data_type)
        return pd.DataFrame()
    
    nato_countries = [
        'United States of America', 'United States', 'Canada', 'United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'Netherlands',
        'Belgium', 'Luxembourg', 'Portugal', 'Denmark', 'Norway', 'Iceland', 'Greece', 'Turkey', 'Turkiye',
        'Poland', 'Czech Republic', 'Czechia', 'Hungary', 'Estonia', 'Latvia', 'Lithuania', 'Slovenia',
        'Slovakia', 'Slovak Republic', 'Bulgaria', 'Romania', 'Croatia', 'Albania', 'Montenegro', 'Macedonia', 'North Macedonia',
        'Finland', 'Sweden'
    ]
    
    additional_countries = ['China', 'Russia', 'Ukraine', 'Belarus', 'Switzerland', 'Japan']
    
    countries_of_interest = nato_countries + additional_countries
    
    df_melted = df.melt(
        id_vars=['Country'], 
        value_vars=year_columns,
        var_name='Year', 
        value_name='Value'
    )
    
    # Extract year from column name based on data type
    if data_type == 'Military Expenditure':
        # For military expenditure, the column name is just the year
        df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
    else:
        # For other data types, extract year from "YYYY_suffix" format
        df_melted['Year'] = df_melted['Year'].str.split('_').str[0]
        df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
    
    # Clean and convert Value column - remove commas before converting to numeric
    df_melted['Value'] = df_melted['Value'].astype(str).str.replace(',', '', regex=False)
    df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')
    
    # Convert to percentage if needed
    if use_percentage:
        df_melted['Value'] = df_melted['Value'] * 100
    
    # Remove rows with missing values
    df_melted = df_melted.dropna(subset=['Value', 'Year'])
    
    df_filtered = df_melted[df_melted['Country'].str.strip().isin(countries_of_interest)]
    
    if df_filtered.empty:
        key_keywords = ['United States', 'Germany', 'France', 'China', 'Russia', 'Ukraine', 'Poland', 'Turkey']
        pattern = '|'.join(key_keywords)
        df_filtered = df_melted[df_melted['Country'].str.contains(pattern, case=False, na=False)]
    
    unique_countries = df_filtered['Country'].unique()
    if len(unique_countries) > 20:
        recent_year = df_filtered['Year'].max()
        country_completeness = df_filtered.groupby('Country').size()
        recent_values = df_filtered[df_filtered['Year'] == recent_year].set_index('Country')['Value']
        
        scores = country_completeness.reindex(unique_countries, fill_value=0) * 0.3 + \
                recent_values.reindex(unique_countries, fill_value=0) * 0.7
        
        top_countries = scores.nlargest(20).index.tolist()
        df_filtered = df_filtered[df_filtered['Country'].isin(top_countries)]
    elif len(unique_countries) < 5:
        recent_year = df_melted['Year'].max()
        top_countries = (df_melted[df_melted['Year'] == recent_year]
                        .nlargest(15, 'Value')['Country'].tolist())
        df_filtered = df_melted[df_melted['Country'].isin(top_countries)]
    
    return df_filtered


def create_racing_bar_chart(data, title, color_scale, use_percentage=False, beginning_year=1995, data_type=None):
    """Create the actual racing bar chart using Plotly"""
    
    if data.empty:
        st.warning("No data available for visualization")
        return

    fig = px.bar(
        data,
        x='Value',
        y='Country',
        animation_frame='Year',
        orientation='h',
        title=title,
        color='Value',
        color_continuous_scale=color_scale,
        range_x=[0, data['Value'].max() * 1.1],
        text='Value',
        height=800,
    )

    text_tmpl = use_percentage and '%{x:.2f}%' or '%{x:,.0f}'
    fig.update_traces(
        texttemplate=text_tmpl,
        textposition='outside',
        textfont_size=18,
    )

    try:
        for frame in fig.frames:
            if hasattr(frame, 'data') and frame.data:
                for trace in frame.data:
                    trace.update(
                        texttemplate=text_tmpl,
                        textposition='outside',
                        textfont_size=18,
                    )
    except (AttributeError, TypeError):
        pass

    years = list(range(beginning_year, 2026))
    steps = []
    if data_type == 'GDP Change From Last Year':
        for yr in years:
            steps.append(dict(
                method="animate",             
                label=str(yr),                
                args=[
                    [str(yr)],             
                    {
                        "mode": "immediate", 
                        "frame": {"duration": 3000, "redraw": False},
                        "transition": {"duration": 200}
                    }
                ]
            ))
    else:
        for yr in years:
            steps.append(dict(
                method="animate",            
                label=str(yr),                
                args=[
                    [str(yr)],              
                    {
                        "mode": "immediate", 
                        "frame": {"duration": 1000, "redraw": False},
                        "transition": {"duration": 300}
                    }
                ]
            ))
    custom_slider = [dict(
        active=0,                        
        currentvalue={"prefix": "Year: ", "font": {"size": 16}},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=custom_slider,
        updatemenus=[],                   
        xaxis_title=(use_percentage and "Percentage (%)" or "Value"),
        yaxis_title="Country",
        yaxis=dict(categoryorder='total ascending', title_font_size=18, tickfont=dict(size=18)),
        font=dict(size=16),
        hoverlabel=dict(font_size=16),
        bargap=0.1,
    )
    hide_slider_rail = """
    <style>
    /* hide the invisible "touch" layer of every Plotly slider */
    .slider-rail-touch-rect {
        display: none !important;
    }
    /* you can also hide the visible rail if you like */
    .slider-rail-rect {
        display: none !important;
    }
    </style>
    """
    st.markdown(hide_slider_rail, unsafe_allow_html=True)

    st.plotly_chart(fig, use_container_width=True)
