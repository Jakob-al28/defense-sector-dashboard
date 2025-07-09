import streamlit as st 
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
import folium
from shapely.geometry import shape, Point
from utils.map_data import load_data, load_geojson
from matplotlib.colors import to_hex

def create_streamlit_legend(bins, colors, title=None):
    title_html = ""
    if title:
        title_html = (
            f"<div style='"
            "font-weight:bold;"
            "font-size:1rem;"
            "margin-bottom:6px;"
            f"'>{title}</div>"
        )
    items = []
    for left, right, col in zip(bins[:-1], bins[1:], colors):
        if isinstance(col, tuple):
            col = to_hex(col)
        items.append(
            f"<div style='"
            "display:flex;"
            "flex-direction:column;"
            "align-items:center;"
            "margin-right:12px;"
            "'>"
            f"<div style='"
            f"background-color:{col};"
            "width:40px;height:20px;"
            "border:1px solid #ccc;"
            "'></div>"
            f"<small style='margin-top:2px;'>{left:.2f} – {right:.2f}</small>"
            "</div>"
        )
    
    return (
        "<div style='display:flex;flex-direction:column;align-items:flex-start;'>"
        f"{title_html}"
        "<div style='display:flex;'>"
        f"{''.join(items)}"
        "</div>"
        "</div>"
        "<br>"
    )

def render_military_expenditure_map(df_military, time=2024):
    try:
        df = df_military
        if df is None:
            return

        geojson_data = load_geojson()
        for feature in geojson_data['features']:
            country_name = feature['properties']['COUNTRY']
            country_data = df[df['Country'] == country_name]
            if not country_data.empty and time in country_data.columns:
                value = country_data[time].iloc[0]
                if pd.notna(value):  
                    feature['properties']['VALUE'] = f"{value * 100:.2f}%"
                else:
                    feature['properties']['VALUE'] = 'N/A'
            else:
                feature['properties']['VALUE'] = 'N/A'
      
        m = folium.Map(location=[45.1109, 20.6821], zoom_start=4, tiles='OpenStreetMap')   

        choropleth = folium.Choropleth(
                geo_data=geojson_data,  
                name='Military Spending',
                data=df,
                columns=['Country', time],
                key_on='feature.properties.COUNTRY',  
                fill_color='RdYlBu_r',
                fill_opacity=0.50,
                line_opacity=0,
                legend_name='Military Spending (% of Govt. Spending)',
                smooth_factor=0,
                bins=[0, 0.02, 0.04, 0.06, 0.1, 0.18, 0.27, 0.36, 0.40, 0.45, 0.60]
            ).add_to(m)

        for key in choropleth._children:
            if key.startswith('color_map'):
                del(choropleth._children[key])

        folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                'fillOpacity': 0,
                'color': 'transparent',
                'weight': 0
            },
            highlight_function=lambda feature: {
                'fillOpacity': 0,
                'color': 'transparent',
                'weight': 0
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['COUNTRY', 'VALUE'],
                aliases=['Country:', 'Military Spending:'],
            )
        ).add_to(m)

        folium.LayerControl().add_to(m)

        output = st_folium(m, width=1200, height=1000, returned_objects=["last_object_clicked"])
        clicked = output.get("last_object_clicked")
        country_name = None
        if clicked:
            pt = Point(clicked["lng"], clicked["lat"])
            country_name = None
            for feature in geojson_data["features"]:
                poly = shape(feature["geometry"])
                if poly.contains(pt):
                    country_name = feature["properties"]["COUNTRY"]
                    break

        bins   = choropleth.color_scale.index   
        colors = choropleth.color_scale.colors 
        legend_html = f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0 auto;
            width: fit-content;
        ">
            <div style="
                text-align: center;
                font-weight: bold;
                margin-bottom: 10px;
                width: 100%;
            ">
                Military Spending (% of Govt. Spending)
            </div>
            {create_streamlit_legend(bins, colors)}
        </div>
        """

        st.markdown(legend_html, unsafe_allow_html=True)
        return country_name
    except Exception as e:
            print(e)

def render_gdp_total_map(df_gdp, time="2024_gdp"):
    try:
        df = df_gdp
        if df is None:
            return
        df[time] = (
            df[time]
              .astype(str)
              .str.replace(',', '', regex=False)
        )
        df[time] = pd.to_numeric(df[time], errors='coerce')

        geojson_data = load_geojson()
        for feature in geojson_data['features']:
            country_name = feature['properties']['COUNTRY']
            country_data = df[df['Country'] == country_name]
            if not country_data.empty and time in country_data.columns:
                value = country_data[time].iloc[0]
                if pd.notna(value) and value != '' and str(value).lower() != 'nan':
                    try:
                        numeric_value = float(value)
                        feature['properties']['VALUE'] = f"{numeric_value:,.1f} B USD"
                    except (ValueError, TypeError):
                        feature['properties']['VALUE'] = 'N/A'
                else:
                    feature['properties']['VALUE'] = 'N/A'
            else:
                feature['properties']['VALUE'] = 'N/A'
        m = folium.Map(location=[45.1109, 20.6821], zoom_start=4, tiles='OpenStreetMap')   

        choropleth = folium.Choropleth(
                geo_data=geojson_data,
                name='GDP',
                data=df,
                columns=['Country', time],
                key_on='feature.properties.COUNTRY',  
                fill_color='RdYlBu',
                fill_opacity=0.50,
                line_opacity=0,
                legend_name='GDP (Billions USD)', 
                smooth_factor=0,
                bins=[0, 1, 2, 200, 500, 1000, 2000, 4500, 20000, 30000]
            ).add_to(m)

        bins   = choropleth.color_scale.index   
        colors = choropleth.color_scale.colors 

        for key in choropleth._children:
            if key.startswith('color_map'):
                del(choropleth._children[key])

        folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                'fillOpacity': 0,
                'color': 'transparent',
                'weight': 0
            },
            highlight_function=lambda feature: {
                'fillOpacity': 0,
                'color': 'transparent',
                'weight': 0
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['COUNTRY', 'VALUE'],
                aliases=['Country:', 'GDP:'],
            )
        ).add_to(m)

        folium.LayerControl().add_to(m)

        output = st_folium(m, width=1200, height=1000, returned_objects=["last_object_clicked"])
        clicked = output.get("last_object_clicked")
        country_name = None
        if clicked:
            pt = Point(clicked["lng"], clicked["lat"])
            country_name = None
            for feature in geojson_data["features"]:
                poly = shape(feature["geometry"])
                if poly.contains(pt):
                    country_name = feature["properties"]["COUNTRY"]
                    break

        bins   = choropleth.color_scale.index   
        colors = choropleth.color_scale.colors 
        legend_html = create_streamlit_legend(bins, colors, "GDP (Billions USD)")

        legend_html = f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0 auto;
            width: fit-content;
        ">
            <div style="
                text-align: center;
                font-weight: bold;
                margin-bottom: 10px;
                width: 100%;
            ">
                GDP (Billions USD)
            </div>
            {create_streamlit_legend(bins, colors)}
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)
        return country_name
    except Exception as e:
            print(e)

def render_gdp_total_map_lin(df_gdp, time="2024_gdp"):
    try:
        df = df_gdp
        if df is None:
            return
        df[time] = (
            df[time]
              .astype(str)
              .str.replace(',', '', regex=False)
        )
        df[time] = pd.to_numeric(df[time], errors='coerce')

        geojson_data = load_geojson()
        for feature in geojson_data['features']:
            country_name = feature['properties']['COUNTRY']
            country_data = df[df['Country'] == country_name]
            if not country_data.empty and time in country_data.columns:
                value = country_data[time].iloc[0]
                if pd.notna(value) and value != '' and str(value).lower() != 'nan':
                    try:
                        numeric_value = float(value)
                        feature['properties']['VALUE'] = f"{numeric_value:,.1f} B USD"
                    except (ValueError, TypeError):
                        feature['properties']['VALUE'] = 'N/A'
                else:
                    feature['properties']['VALUE'] = 'N/A'
            else:
                feature['properties']['VALUE'] = 'N/A'
        m = folium.Map(location=[45.1109, 20.6821], zoom_start=4, tiles='OpenStreetMap')   

        choropleth = folium.Choropleth(
                geo_data=geojson_data,
                name='GDP',
                data=df,
                columns=['Country', time],
                key_on='feature.properties.COUNTRY',  
                fill_color='RdYlBu_r',
                fill_opacity=0.50,
                line_opacity=0,
                legend_name='GDP (Billions USD)', 
                smooth_factor=0,
            ).add_to(m)

        bins   = choropleth.color_scale.index   
        colors = choropleth.color_scale.colors 

        for key in choropleth._children:
            if key.startswith('color_map'):
                del(choropleth._children[key])

        folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                'fillOpacity': 0,
                'color': 'transparent',
                'weight': 0
            },
            highlight_function=lambda feature: {
                'fillOpacity': 0,
                'color': 'transparent',
                'weight': 0
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['COUNTRY', 'VALUE'],
                aliases=['Country:', 'GDP:'],
            )
        ).add_to(m)

        folium.LayerControl().add_to(m)

        output = st_folium(m, width=1200, height=1000, returned_objects=["last_object_clicked"])
        clicked = output.get("last_object_clicked")
        country_name = None
        if clicked:
            pt = Point(clicked["lng"], clicked["lat"])
            country_name = None
            for feature in geojson_data["features"]:
                poly = shape(feature["geometry"])
                if poly.contains(pt):
                    country_name = feature["properties"]["COUNTRY"]
                    break        
        bins   = choropleth.color_scale.index   
        colors = choropleth.color_scale.colors 
        legend_html = create_streamlit_legend(bins, colors, "GDP (Billions USD)")
        legend_html = f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0 auto;
            width: fit-content;
        ">
            <div style="
                text-align: center;
                font-weight: bold;
                margin-bottom: 10px;
                width: 100%;
            ">
                GDP (Billions USD)
            </div>
            {create_streamlit_legend(bins, colors)}
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)
        return country_name
    except Exception as e:
            print(e)

def render_gdp_total_map_log(df_gdp, time="2024_gdp"):
    try:
        df = df_gdp.copy()
        if df is None or df.empty:
            return

        df[time] = (
            df[time]
              .astype(str)
              .str.replace(',', '', regex=False)
        )
        df[time] = pd.to_numeric(df[time], errors='coerce')

        floor = 1e-3
        df['log_gdp'] = np.log10(df[time].clip(lower=floor))

        geojson_data = load_geojson()
        for feat in geojson_data['features']:
            country = feat['properties']['COUNTRY']
            val = df.loc[df['Country'] == country, time]
            feat['properties']['VALUE'] = (
                f"{val.iloc[0]:,.1f} B USD"
                if not val.empty and pd.notna(val.iloc[0])
                else "N/A"
            )

        valid = df['log_gdp'].dropna()
        if valid.empty:
            return
        min_log, max_log = valid.min(), valid.max()
        if min_log == max_log:
            min_log -= 0.1; max_log += 0.1
        log_bins = np.linspace(min_log, max_log, num=6)

        m = folium.Map(location=[45.1109, 20.6821], zoom_start=4, tiles='OpenStreetMap')

        choropleth = folium.Choropleth(
            geo_data=geojson_data,
            name='GDP (log₁₀ scale)',
            data=df,
            columns=['Country', 'log_gdp'],
            key_on='feature.properties.COUNTRY',
            fill_color='RdYlBu',
            bins=log_bins,              
            fill_opacity=0.5,
            line_opacity=0,
            legend_name='log₁₀(GDP in B USD)',
            smooth_factor=0
        ).add_to(m)

        for key in list(choropleth._children):
            if key.startswith('color_map'):
                del choropleth._children[key]

        folium.GeoJson(
            geojson_data,
            style_function=lambda feat: {'fillOpacity': 0, 'color': 'transparent', 'weight': 0},
            highlight_function=lambda feat: {'fillOpacity': 0, 'color': 'transparent', 'weight': 0},
            tooltip=folium.GeoJsonTooltip(
                fields=['COUNTRY','VALUE'],
                aliases=['Country:','GDP:'],
            )
        ).add_to(m)

        folium.LayerControl().add_to(m)
        output = st_folium(m, width=1200, height=1000, returned_objects=["last_object_clicked"])
        clicked = output.get("last_object_clicked")
        country_name = None
        if clicked:
            pt = Point(clicked["lng"], clicked["lat"])
            country_name = None
            for feature in geojson_data["features"]:
                poly = shape(feature["geometry"])
                if poly.contains(pt):
                    country_name = feature["properties"]["COUNTRY"]
                    break
        bins   = choropleth.color_scale.index   
        colors = choropleth.color_scale.colors 
        legend_html = create_streamlit_legend(bins, colors, "log₁₀(GDP in B USD)")
        legend_html = f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0 auto;
            width: fit-content;
        ">
            <div style="
                text-align: center;
                font-weight: bold;
                margin-bottom: 10px;
                width: 100%;
            ">
                log₁₀(GDP in B USD)
            </div>
            {create_streamlit_legend(bins, colors)}
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)
        return country_name
    except Exception as e:
        print("Error rendering log‐scale GDP map:", e)

def render_gdp_change_map(df_gdp_change, time="2024_gdp_delta"):
    try:
        df = df_gdp_change
        if df is None:
            return
        df[time] = (
            df[time]
              .astype(str)
              .str.replace(',', '', regex=False)
        )
        df[time] = pd.to_numeric(df[time], errors='coerce')

        geojson_data = load_geojson()
        for feature in geojson_data['features']:
            country_name = feature['properties']['COUNTRY']
            country_data = df[df['Country'] == country_name]
            if not country_data.empty and time in country_data.columns:
                value = country_data[time].iloc[0]
                if pd.notna(value) and value != '' and str(value).lower() != 'nan':
                    try:
                        numeric_value = float(value)
                        feature['properties']['VALUE'] = f"{numeric_value:,.1f} %"
                    except (ValueError, TypeError):
                        feature['properties']['VALUE'] = 'N/A'
                else:
                    feature['properties']['VALUE'] = 'N/A'
            else:
                feature['properties']['VALUE'] = 'N/A'
        # Create base map
        m = folium.Map(location=[45.1109, 20.6821], zoom_start=4, tiles='OpenStreetMap')   

        # Add choropleth layer
        choropleth = folium.Choropleth(
                geo_data=geojson_data,
                name='GDP',
                data=df,
                columns=['Country', time],
                key_on='feature.properties.COUNTRY',  
                fill_color='RdYlBu',
                fill_opacity=0.50,
                line_opacity=0,
                legend_name='GDP (Billions USD)', 
                smooth_factor=0,
                bins=[-15, -2, 0, 2, 4, 8, 50]
            ).add_to(m)

        for key in choropleth._children:
            if key.startswith('color_map'):
                del(choropleth._children[key])

        folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                'fillOpacity': 0,
                'color': 'transparent',
                'weight': 0
            },
            highlight_function=lambda feature: {
                'fillOpacity': 0,
                'color': 'transparent',
                'weight': 0
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['COUNTRY', 'VALUE'],
                aliases=['Country:', 'GDP %:'],
            )
        ).add_to(m)

        folium.LayerControl().add_to(m)

        #st_folium(m, width=725, height=500, returned_objects=[])
        output = st_folium(m, width=1200, height=1000, returned_objects=["last_object_clicked"])
        clicked = output.get("last_object_clicked")
        country_name = None
        if clicked:
            pt = Point(clicked["lng"], clicked["lat"])
            country_name = None
            for feature in geojson_data["features"]:
                poly = shape(feature["geometry"])
                if poly.contains(pt):
                    country_name = feature["properties"]["COUNTRY"]
                    break        
        bins   = choropleth.color_scale.index   
        colors = choropleth.color_scale.colors 
        legend_html = create_streamlit_legend(bins, colors, "GDP % Change")
        legend_html = f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0 auto;
            width: fit-content;
        ">
            <div style="
                text-align: center;
                font-weight: bold;
                margin-bottom: 10px;
                width: 100%;
            ">
                GDP % Change
            </div>
            {create_streamlit_legend(bins, colors)}
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)
        return country_name
    except Exception as e:
            print(e)

def render_gdp_per_capita_total_map(df_gdppc, time="2024_gdppc"):
    # Load GDP data
    try:
        df = df_gdppc
        if df is None:
            return
        df[time] = (
            df[time]
              .astype(str)
              .str.replace(',', '', regex=False)
        )
        df[time] = pd.to_numeric(df[time], errors='coerce')

        geojson_data = load_geojson()
        for feature in geojson_data['features']:
            country_name = feature['properties']['COUNTRY']
            country_data = df[df['Country'] == country_name]
            if not country_data.empty and time in country_data.columns:
                value = country_data[time].iloc[0]
                if pd.notna(value) and value != '' and str(value).lower() != 'nan':
                    try:
                        numeric_value = float(value)
                        feature['properties']['VALUE'] = f"{numeric_value:,.1f} USD"
                    except (ValueError, TypeError):
                        feature['properties']['VALUE'] = 'N/A'
                else:
                    feature['properties']['VALUE'] = 'N/A'
            else:
                feature['properties']['VALUE'] = 'N/A'
        # Create base map
        m = folium.Map(location=[45.1109, 20.6821], zoom_start=4, tiles='OpenStreetMap')   

        # Add choropleth layer
        choropleth = folium.Choropleth(
                geo_data=geojson_data,
                name='GDP',
                data=df,
                columns=['Country', time],
                key_on='feature.properties.COUNTRY',  
                fill_color='RdYlBu',
                fill_opacity=0.50,
                line_opacity=0,
                legend_name='GDP (Billions USD)', 
                smooth_factor=0,
                bins=[0, 1000, 2500, 5000, 7500, 10000, 20000, 30000, 40000, 50000, 60000, 80000, 100000, 150000]
            ).add_to(m)

        for key in choropleth._children:
            if key.startswith('color_map'):
                del(choropleth._children[key])

        folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                'fillOpacity': 0,
                'color': 'transparent',
                'weight': 0
            },
            highlight_function=lambda feature: {
                'fillOpacity': 0,
                'color': 'transparent',
                'weight': 0
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['COUNTRY', 'VALUE'],
                aliases=['Country:', 'GDP Per Capita:'],
            )
        ).add_to(m)

        folium.LayerControl().add_to(m)

        #st_folium(m, width=725, height=500, returned_objects=[])
        output = st_folium(m, width=1200, height=1000, returned_objects=["last_object_clicked"])
        clicked = output.get("last_object_clicked")
        country_name = None
        if clicked:
            pt = Point(clicked["lng"], clicked["lat"])
            country_name = None
            for feature in geojson_data["features"]:
                poly = shape(feature["geometry"])
                if poly.contains(pt):
                    country_name = feature["properties"]["COUNTRY"]
                    break
        bins   = choropleth.color_scale.index   
        colors = choropleth.color_scale.colors 
        legend_html = create_streamlit_legend(bins, colors, "GDP Per Capita (USD)")
        legend_html = f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0 auto;
            width: fit-content;
        ">
            <div style="
                text-align: center;
                font-weight: bold;
                margin-bottom: 10px;
                width: 100%;
            ">
                GDP Per Capita (USD)
            </div>
            {create_streamlit_legend(bins, colors)}
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)
        return country_name

    except Exception as e:
            print(e)

def render_gdp_total_per_capita_map_log(df_gdppc, time="2024_gdppc"):
    """Render GDP total values map on a log scale."""
    try:
        df = df_gdppc.copy()
        if df is None or df.empty:
            return

        df[time] = (
            df[time]
              .astype(str)
              .str.replace(',', '', regex=False)
        )
        df[time] = pd.to_numeric(df[time], errors='coerce')

        floor = 1e-3
        df['log_gdp'] = np.log10(df[time].clip(lower=floor))

        geojson_data = load_geojson()
        for feat in geojson_data['features']:
            country = feat['properties']['COUNTRY']
            val = df.loc[df['Country'] == country, time]
            feat['properties']['VALUE'] = (
                f"{val.iloc[0]:,.1f} USD"
                if not val.empty and pd.notna(val.iloc[0])
                else "N/A"
            )

        valid = df['log_gdp'].dropna()
        if valid.empty:
            return
        min_log, max_log = valid.min(), valid.max()
        if min_log == max_log:
            min_log -= 0.1; max_log += 0.1
        log_bins = np.linspace(min_log, max_log, num=6)

        m = folium.Map(location=[45.1109, 20.6821], zoom_start=4, tiles='OpenStreetMap')

        choropleth = folium.Choropleth(
            geo_data=geojson_data,
            name='GDP (log₁₀ scale)',
            data=df,
            columns=['Country', 'log_gdp'],
            key_on='feature.properties.COUNTRY',
            fill_color='RdYlBu',
            bins=log_bins,              
            fill_opacity=0.5,
            line_opacity=0,
            legend_name='log₁₀(GDP in B USD)',
            smooth_factor=0
        ).add_to(m)

        for key in list(choropleth._children):
            if key.startswith('color_map'):
                del choropleth._children[key]

        folium.GeoJson(
            geojson_data,
            style_function=lambda feat: {'fillOpacity': 0, 'color': 'transparent', 'weight': 0},
            highlight_function=lambda feat: {'fillOpacity': 0, 'color': 'transparent', 'weight': 0},
            tooltip=folium.GeoJsonTooltip(
                fields=['COUNTRY','VALUE'],
                aliases=['Country:','GDP:'],
            )
        ).add_to(m)


        folium.LayerControl().add_to(m)
        #st_folium(m, width=725, height=500, returned_objects=[])
        output = st_folium(m, width=1200, height=1000, returned_objects=["last_object_clicked"])
        clicked = output.get("last_object_clicked")
        country_name = None
        if clicked:
            pt = Point(clicked["lng"], clicked["lat"])
            country_name = None
            for feature in geojson_data["features"]:
                poly = shape(feature["geometry"])
                if poly.contains(pt):
                    country_name = feature["properties"]["COUNTRY"]
                    break
        bins   = choropleth.color_scale.index   
        colors = choropleth.color_scale.colors 
        legend_html = create_streamlit_legend(bins, colors, "log₁₀ GDP Per Capita (USD)")
        legend_html = f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0 auto;
            width: fit-content;
        ">
            <div style="
                text-align: center;
                font-weight: bold;
                margin-bottom: 10px;
                width: 100%;
            ">
                log₁₀ GDP Per Capita (USD)
            </div>
            {create_streamlit_legend(bins, colors)}
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)
        return country_name
    except Exception as e:
        print("Error rendering log‐scale GDP map:", e)

def render_peace_index_map(df_peace, time = "Total_peace"):
    # Load GDP data
    try:
        df = df_peace
        if df is None:
            return
        df[time] = (
            df[time]
              .astype(str)
              .str.replace(',', '', regex=False)
        )
        df[time] = pd.to_numeric(df[time], errors='coerce')

        geojson_data = load_geojson()
        for feature in geojson_data['features']:
            country_name = feature['properties']['COUNTRY']
            # Find matching country in DataFrame
            country_data = df[df['Country'] == country_name]
            if not country_data.empty and time in country_data.columns:
                value = country_data[time].iloc[0]
                # Fix: Handle different data types properly
                if pd.notna(value) and value != '' and str(value).lower() != 'nan':
                    try:
                        numeric_value = float(value)
                        feature['properties']['VALUE'] = f"{numeric_value:,.1f}"
                    except (ValueError, TypeError):
                        feature['properties']['VALUE'] = 'N/A'
                else:
                    feature['properties']['VALUE'] = 'N/A'
            else:
                feature['properties']['VALUE'] = 'N/A'
        # Create base map
        m = folium.Map(location=[45.1109, 20.6821], zoom_start=4, tiles='OpenStreetMap')   

        # Add choropleth layer
        choropleth = folium.Choropleth(
                geo_data=geojson_data,
                name='GDP',
                data=df,
                columns=['Country', time],
                key_on='feature.properties.COUNTRY',  
                fill_color='RdYlBu_r',
                fill_opacity=0.50,
                line_opacity=0,
                legend_name='GDP (Billions USD)',  
                smooth_factor=0,
            ).add_to(m)

        for key in choropleth._children:
            if key.startswith('color_map'):
                del(choropleth._children[key])

        folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                'fillOpacity': 0,
                'color': 'transparent',
                'weight': 0
            },
            highlight_function=lambda feature: {
                'fillOpacity': 0,
                'color': 'transparent',
                'weight': 0
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['COUNTRY', 'VALUE'],
                aliases=['Country:', 'Peace Index:'],
            )
        ).add_to(m)

        folium.LayerControl().add_to(m)

        #st_folium(m, width=725, height=500, returned_objects=[])
        output = st_folium(m, width=1200, height=1000, returned_objects=["last_object_clicked"])
        clicked = output.get("last_object_clicked")
        country_name = None
        if clicked:
            pt = Point(clicked["lng"], clicked["lat"])
            country_name = None
            for feature in geojson_data["features"]:
                poly = shape(feature["geometry"])
                if poly.contains(pt):
                    country_name = feature["properties"]["COUNTRY"]
                    break        
        bins   = choropleth.color_scale.index   
        colors = choropleth.color_scale.colors 
        legend_html = create_streamlit_legend(bins, colors, "Peace Index")
        legend_html = f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0 auto;
            width: fit-content;
        ">
            <div style="
                text-align: center;
                font-weight: bold;
                margin-bottom: 10px;
                width: 100%;
            ">
                Peace Index
            </div>
            {create_streamlit_legend(bins, colors)}
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)
        return country_name
    except Exception as e:
            print(e)

def render_co2_emissions_map(df_sus, time="2023_co2"):
    """Render CO2 emissions map"""
    # Load GDP data
    try:
        df = df_sus
        if df is None:
            return
        df[time] = (
            df[time]
              .astype(str)
              .str.replace(',', '', regex=False)
        )
        df[time] = pd.to_numeric(df[time], errors='coerce')

        geojson_data = load_geojson()
        for feature in geojson_data['features']:
            country_name = feature['properties']['COUNTRY']
            # Find matching country in DataFrame
            country_data = df[df['Country'] == country_name]
            if not country_data.empty and time in country_data.columns:
                value = country_data[time].iloc[0]
                # Fix: Handle different data types properly
                if pd.notna(value) and value != '' and str(value).lower() != 'nan':
                    try:
                        numeric_value = float(value)
                        feature['properties']['VALUE'] = f"{numeric_value:,.1f} B"
                    except (ValueError, TypeError):
                        feature['properties']['VALUE'] = 'N/A'
                else:
                    feature['properties']['VALUE'] = 'N/A'
            else:
                feature['properties']['VALUE'] = 'N/A'
        # Create base map
        m = folium.Map(location=[45.1109, 20.6821], zoom_start=4, tiles='OpenStreetMap')   

        # Add choropleth layer
        choropleth = folium.Choropleth(
                geo_data=geojson_data,
                name='GDP',
                data=df,
                columns=['Country', time],
                key_on='feature.properties.COUNTRY',  
                fill_color='RdYlBu_r',
                fill_opacity=0.50,
                line_opacity=0,
                legend_name='GDP (Billions USD)',  
                smooth_factor=0,
                bins=[0, 100, 200, 400, 1000, 2000, 3000, 6000, 14000]
            ).add_to(m)

        for key in choropleth._children:
            if key.startswith('color_map'):
                del(choropleth._children[key])

        folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                'fillOpacity': 0,
                'color': 'transparent',
                'weight': 0
            },
            highlight_function=lambda feature: {
                'fillOpacity': 0,
                'color': 'transparent',
                'weight': 0
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['COUNTRY', 'VALUE'],
                aliases=['Country:', 'Peace Index:'],
            )
        ).add_to(m)

        folium.LayerControl().add_to(m)

        #st_folium(m, width=725, height=500, returned_objects=[])
        output = st_folium(m, width=1200, height=1000, returned_objects=["last_object_clicked"])
        clicked = output.get("last_object_clicked")
        country_name = None
        if clicked:
            pt = Point(clicked["lng"], clicked["lat"])
            country_name = None
            for feature in geojson_data["features"]:
                poly = shape(feature["geometry"])
                if poly.contains(pt):
                    country_name = feature["properties"]["COUNTRY"]
                    break
        bins   = choropleth.color_scale.index   
        colors = choropleth.color_scale.colors 
        legend_html = create_streamlit_legend(bins, colors, "Co2 Emissions (Billion Tonnes)")
        legend_html = f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0 auto;
            width: fit-content;
        ">
            <div style="
                text-align: center;
                font-weight: bold;
                margin-bottom: 10px;
                width: 100%;
            ">
                Co2 Emissions (Billion Tonnes)
            </div>
            {create_streamlit_legend(bins, colors)}
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)
        return country_name
    except Exception as e:
            print(e)

def render_map():
    st.markdown("""
        <style>
        /* Force everything to full opacity */
        [data-baseweb="tab-panel"],
        [data-baseweb="tab-panel"] *,
        .css-1vencpc, .css-1vencpc * {
            opacity: 100% !important;
            transition: none !important;
        /* Fix for streamlit-folium empty space issue */
        iframe[title="streamlit_folium.st_folium"] { 
            height: 800px !important;
        }
        </style>
    """, unsafe_allow_html=True)


    st.subheader("Current Global Economic and Military Trend Map")
    st.info("**Note:** The scale was intentionally made non-linear to better compare countries with each other.")
    data_type = st.selectbox(
        "Choose data to visualize:",
        options=["Military Expenditure", "GDP Total", "GDP per capita", "GDP Change From Last Year", "Peace Index", "CO2 Emissions"],
        index=0, 
    )
    df = load_data()
    df = df.drop(114)
    df_military = df.iloc[:, :38]
    df_gdp = df[['Country']].join(df.iloc[:, 38:83])
    df_gdp_change = df[['Country']].join(df.iloc[:, 83:93])
    df_emissions = df[['Country']].join(df.iloc[:, 93:117])
    df_peace = df[['Country', df.columns[117]]]    
    df_gdppc = df[['Country']].join(df.iloc[:, 117:])
    if data_type == "Military Expenditure":
        st.write("**Military Expenditure**: Shows military spending as a percentage of total government spending by country in 2024.")
    elif data_type == "GDP Total":
        st.write("**GDP Total**: Shows the absolute Gross Domestic Product values in USD billions for 2024.")
        scale_option = st.radio(
            "Select scaling option:",
            options=["Logarithmic", "Linear"],
            index=0,  
            key="gdp_scale_option"
        )
    elif data_type == "GDP per capita":
        st.write("**GDP per capita**: Shows the GDP per capita in USD for 2024.")
        log_scale2 = st.checkbox("Enable logarithmic scaling", key="log_scale", value=False)
    elif data_type == "GDP Change From Last Year":
        st.write("**GDP Change From Last Year**: Shows the percentage change in GDP for 2024 year.")
    elif data_type == "Peace Index":
        st.write("**Peace Index**: Shows the Fragile States Index, higher values indicate more fragile states in 2024.")
    elif data_type == "CO2 Emissions":
        st.write("**CO2 Emissions**: Shows total greenhouse gas emissions measured in billion tonnes in 2024.")
    try:
        if data_type == "Military Expenditure":
            country_name = render_military_expenditure_map(df_military)
        elif data_type == "GDP Total":
            if scale_option == "Linear":
                country_name =render_gdp_total_map_lin(df_gdp)
            elif scale_option == "Logarithmic":
                country_name =render_gdp_total_map_log(df_gdp)
        elif data_type == "GDP per capita" and not log_scale2:
            country_name =render_gdp_per_capita_total_map(df_gdppc)
        elif data_type == "GDP per capita" and log_scale2:
            country_name =render_gdp_total_per_capita_map_log(df_gdppc)
        elif data_type == "GDP Change From Last Year":
            country_name =render_gdp_change_map(df_gdp_change)
        elif data_type == "Peace Index":
            country_name =render_peace_index_map(df_peace)
        elif data_type == "CO2 Emissions":
            country_name =render_co2_emissions_map(df_emissions)
        if 'clicked_countries' not in st.session_state:
            st.session_state.clicked_countries = []
        if country_name or st.session_state.clicked_countries:
            if country_name and country_name not in st.session_state.clicked_countries:
                st.session_state.clicked_countries.append(country_name)
        return data_type

    except Exception as e:
        pass
