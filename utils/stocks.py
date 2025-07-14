import streamlit as st
import pandas as pd
import datetime
import yfinance as yf
from backend import data_fetching
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
import plotly.graph_objects as go


def fuzzy_search(query, items, threshold=2):
    """Simple fuzzy search using Levenshtein distance"""
    if not query:
        return items
    
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    matches = []
    for item in items:
        ticker_distance = levenshtein_distance(query.lower(), item["ticker"].lower())
        name_distance = levenshtein_distance(query.lower(), item["name"].lower())
        
        min_distance = min(ticker_distance, name_distance)
        if min_distance <= threshold:
            matches.append((item, min_distance))
    
    return [item for item, _ in sorted(matches, key=lambda x: x[1])]


def render_stocks():
    if 'rows_per_page' not in st.session_state:
        st.session_state.rows_per_page = 25
    if 'search_filter' not in st.session_state:
        st.session_state.search_filter = ""
    if 'group_filter' not in st.session_state:
        st.session_state.group_filter = "All Stocks"
    if 'sort_column' not in st.session_state:
        st.session_state.sort_column = "ticker"
    if 'sort_direction' not in st.session_state:
        st.session_state.sort_direction = "asc"
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = None
    if 'show_stock_modal' not in st.session_state:
        st.session_state.show_stock_modal = False
    if 'first_load_complete' not in st.session_state:
        st.session_state.first_load_complete = True
        st.session_state.selected_stock = "RHM.F"  
        st.session_state.show_stock_modal = True

    sample_quotes = [
        {"ticker": "AAPL", "name": "Apple Inc.", "last_price": 185.92, "change": 2.45, "change_pct": 1.34, "volume": 1250000, "bid": 185.90, "ask": 185.95, "day_high": 186.45, "day_low": 184.20, "year_high": 198.23, "year_low": 124.17, "market_cap": "2.89T"},
        {"ticker": "MSFT", "name": "Microsoft Corp.", "last_price": 378.85, "change": -1.23, "change_pct": -0.32, "volume": 890000, "bid": 378.80, "ask": 378.90, "day_high": 379.50, "day_low": 377.20, "year_high": 420.00, "year_low": 280.00, "market_cap": "2.81T"},
        {"ticker": "GOOGL", "name": "Alphabet Inc.", "last_price": 142.56, "change": 0.78, "change_pct": 0.55, "volume": 2100000, "bid": 142.50, "ask": 142.60, "day_high": 143.20, "day_low": 141.80, "year_high": 150.00, "year_low": 120.00, "market_cap": "1.79T"},
        {"ticker": "AMZN", "name": "Amazon.com Inc.", "last_price": 145.24, "change": -3.12, "change_pct": -2.10, "volume": 3400000, "bid": 145.20, "ask": 145.30, "day_high": 148.50, "day_low": 144.80, "year_high": 160.00, "year_low": 100.00, "market_cap": "1.51T"},
        {"ticker": "TSLA", "name": "Tesla Inc.", "last_price": 248.50, "change": 12.45, "change_pct": 5.28, "volume": 5600000, "bid": 248.40, "ask": 248.60, "day_high": 250.00, "day_low": 240.00, "year_high": 300.00, "year_low": 150.00, "market_cap": "789B"},
        {"ticker": "META", "name": "Meta Platforms Inc.", "last_price": 334.69, "change": 5.67, "change_pct": 1.72, "volume": 1800000, "bid": 334.60, "ask": 334.80, "day_high": 335.50, "day_low": 330.00, "year_high": 380.00, "year_low": 200.00, "market_cap": "851B"},
        {"ticker": "NVDA", "name": "NVIDIA Corp.", "last_price": 485.09, "change": 15.23, "change_pct": 3.24, "volume": 4200000, "bid": 485.00, "ask": 485.20, "day_high": 490.00, "day_low": 480.00, "year_high": 500.00, "year_low": 300.00, "market_cap": "1.20T"},
        {"ticker": "NFLX", "name": "Netflix Inc.", "last_price": 492.98, "change": -8.45, "change_pct": -1.68, "volume": 950000, "bid": 492.90, "ask": 493.10, "day_high": 500.00, "day_low": 490.00, "year_high": 550.00, "year_low": 400.00, "market_cap": "218B"},
    ]

    st.markdown("### Price Grid")

    col1, col2 = st.columns(2)
    with col1:
        search_filter = st.text_input(
            "Search (Ticker/Name)",
            placeholder="AAPL, Apple...",
            key="search_control"
        )
    with col2:
        group_options = [
            "Aerospace & Defense",
            "ESG ETFs",
            "All Stocks",
            "Technology", 
            "Financial Services",
            "Consumer Discretionary",
            "Communication Services",
            "Industrials",
            "Consumer Staples",
            "Energy",
            "Materials",
            "ETFs",
        ]
        
        group_filter = st.selectbox(
            "Filter by Sector/Group",
            group_options,
            index=0,  
            key="group_filter_control"
        )

    group_mapping = {
        "Aerospace & Defense": [
            "RTX", "PPA", "RHM.F", "HON", "SAF.PA", "LMT", "BA.L","HO.PA",
            "HAL.NS", "BEL.NS", "KOG.OL", "LDO.MI", "SAAB-B.ST", "AM.PA", "601985.SS",
            "600760.SS", "ASELS.IS", "HAG.F",
            # ETF
            "XAR", "ITA", "SHLD", "JETS"
        ],
        "ESG ETFs": [
            "VSGX", "DSI", "EAGG", "ESGU", "ESGD", "ESGE", "CHGX", "FGDL", "CXSE"
        ],
        "Technology": [
            "AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ORCL", "INTC", "IBM", "CSCO", "SAP"
        ],
        "Financial Services": [
            "JPM", "BAC", "WFC", "MS", "GS", "C", "SCHW", "BLK"
        ],
        # "Healthcare": [
        #     "UNH", "HCA", "MCK", "CI", "ELV", "PFE", "LLY"
        # ],
        "Consumer Discretionary": [
            "AMZN", "TSLA", "HD", "MCD", "BKNG", "TM", "SONY"
        ],
        "Communication Services": [
            "GOOGL", "META", "NFLX", "TMUS", "DIS", "VZ", "T", "CMCSA", "ABNB", "DASH"
        ],
        "Industrials": [
            "GE", "RTX", "CAT", "BA", "HON", "ETN", "UNP"
        ],
        "Consumer Staples": [
            "WMT", "COST", "PG", "KO", "PEP", "PM", "MO"
        ],
        "Energy": [
            "2222.SR", "XOM", "CVX", "SHEL", "0857.HK"
        ],
        # "Real Estate": [
        #     "AMT", "PLD", "WELL", "EQIX"
        # ],
        "Materials": [
            "LIN", "SHW", "APD", "SCCO", "ECL", "FCX", "NEM"
        ],
        "ETFs": [
            "SPY", "IVV", "VOO", "VTI", "QQQ", "IWM", "DIA", "EEM", "GLD", "AGG"
        ]
    }

    if group_filter == "All Stocks":
        all_tickers = []
        for ticker_list in group_mapping.values():
            all_tickers.extend(ticker_list)
        quotes_df = data_fetching.get_top_stocks_quotes(all_tickers)
    else:
        tickers_to_fetch = group_mapping.get(group_filter, [])
        if tickers_to_fetch:
            quotes_df = data_fetching.get_top_stocks_quotes(tickers=tickers_to_fetch)
        else:
            st.warning(f"No stocks defined for group: {group_filter}")
            all_tickers = []
            for ticker_list in group_mapping.values():
                all_tickers.extend(ticker_list)
            quotes_df = data_fetching.get_top_stocks_quotes(all_tickers)
    
    if not isinstance(quotes_df, pd.DataFrame):
        st.error("Error loading stock data")
        return
    
    quotes_df = quotes_df.drop(columns=['open'], errors='ignore')

    if search_filter:
        search = search_filter.lower()
        quotes_df = quotes_df[
            quotes_df['ticker'].str.lower().str.contains(search) |
            quotes_df['name'].str.lower().str.contains(search)
        ]

    st.markdown("**Click on the stock to view details**")
    columns_to_drop = ['currency', 'bid', 'ask', 'year_high', 'year_low', 'market_cap']
    display_df = quotes_df.drop(columns=[col for col in columns_to_drop if col in quotes_df.columns], errors='ignore')
    
    if not isinstance(display_df, pd.DataFrame):
        st.error("Error processing stock data for display")
        return
        
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=11)
    gb.configure_default_column(editable=False, groupable=False)
    gb.configure_selection('single', use_checkbox=False)

    gb.configure_column(
        "last_price",
        header_name="Last Price",
        type=["numericColumn"],
        valueFormatter="`$${x.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`",
        cellStyle={"color": "#fafafa"}
    )
    gb.configure_column(
        "day_high",
        header_name="Day High",
        type=["numericColumn"],
        valueFormatter="`$${x.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`",
        cellStyle={"color": "#2ca02c"}
    )
    gb.configure_column(
        "day_low",
        header_name="Day Low",
        type=["numericColumn"],
        valueFormatter="`$${x.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`",
        cellStyle={"color": "#d62728"}
    )
    gb.configure_column(
        "change", 
        header_name="Change", 
        type=["numericColumn"], 
        valueFormatter="x.toFixed(2)",
        cellStyle=JsCode("""
            function(params) {
                if (params.value > 0) {
                    return {'color': '#2ca02c', 'fontWeight': 'bold'};
                } else if (params.value < 0) {
                    return {'color': '#d62728', 'fontWeight': 'bold'};
                } else {
                    return {'color': '#fafafa'};
                }
            }
        """)
    )
    gb.configure_column(
        "change_pct", 
        header_name="% Change", 
        type=["numericColumn"], 
        valueFormatter="x.toFixed(2) + '%'",
        cellStyle=JsCode("""
            function(params) {
                if (params.value > 0) {
                    return {'color': '#2ca02c', 'fontWeight': 'bold'};
                } else if (params.value < 0) {
                    return {'color': '#d62728', 'fontWeight': 'bold'};
                } else {
                    return {'color': '#fafafa', 'backgroundColor': '#f5f5f5'};
                }
            }
        """)
    )
    gb.configure_column("ticker", header_name="Ticker", cellStyle={"color": "#fafafa"})
    gb.configure_column("name", header_name="Company", cellStyle={"color": "#fafafa"})
    gb.configure_column("volume", header_name="Volume", cellStyle={"color": "#fafafa"})

    gridOptions = gb.build()
    gridOptions['enableRangeSelection'] = True
    gridOptions['enableCellTextSelection'] = True

    custom_css = {
        ".ag-header-cell": {
            "color": "#fafafa !important",
            "font-weight": "bold !important",
            "border-bottom": "2px solid #262730 !important"
        },
        ".ag-row": {
            "border-bottom": "1px solid #262730 !important"
        },
        ".ag-row:hover": {
            "background-color": "#0e1117 !important"
        },
        ".ag-cell": {
            "font-family": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important",
            "font-size": "14px !important"
        },
        ".ag-row-selected": {
            "background-color": "#0e1117 !important"
        }
    }

    response = AgGrid(
        display_df,
        gridOptions=gridOptions,
        update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        theme='streamlit',
        fit_columns_on_grid_load=True,
        height=400,
        custom_css=custom_css
    )
    selected_rows = response['selected_rows']
    if isinstance(selected_rows, pd.DataFrame) and len(selected_rows) > 0:
        selected_ticker = selected_rows.iloc[0]['ticker']
        st.session_state.selected_stock = selected_ticker
        st.session_state.show_stock_modal = True
    else:
        pass
        

    st.session_state.search_filter = search_filter
    st.session_state.group_filter = group_filter
    
    st.session_state.rows_per_page = 5
    
    
    filtered_quotes = sample_quotes
    if st.session_state.search_filter:
        filtered_quotes = fuzzy_search(st.session_state.search_filter, sample_quotes, threshold=2)
    
    if st.session_state.search_filter and len(filtered_quotes) < len(sample_quotes):
        st.caption(f"Found {len(filtered_quotes)} matches for '{st.session_state.search_filter}'")
    st.caption(f"Data source: Yahoo Finance | Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")    
    st.markdown("---")
    
    # Detail Modal
    if st.session_state.show_stock_modal and st.session_state.selected_stock:
        selected_row = quotes_df[quotes_df['ticker'] == st.session_state.selected_stock]
        selected_stock_data = None
        if isinstance(selected_row, pd.DataFrame) and not selected_row.empty:
            selected_stock_data = selected_row.iloc[0]
        
        if selected_stock_data is not None:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"### {selected_stock_data['ticker']} - {selected_stock_data['name']} (Detail View)")
                with col2:
                    if st.button("Close", key="close_modal"):
                        st.session_state.show_stock_modal = False
                        st.session_state.selected_stock = None
                        st.rerun()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    chart_type = st.selectbox(
                        "Chart Type:",
                        ["Line Chart", "Candlestick Chart"],
                        index=0,  
                        key="chart_type"
                    )
                
                with col2:
                    interval_map = {
                        "1 minute": "1m",
                        "5 minutes": "5m",
                        "15 minutes": "15m",
                        "1 hour": "1h",
                        "1 day": "1d",
                    }
                    interval_label = st.selectbox(
                        "Time Interval:",
                        list(interval_map.keys()),
                        index=4, 
                        key="time_interval"
                    )
                    interval = interval_map[interval_label]
                    
                with col3:
                    # Time range selector with limitations based on interval
                    if interval in ['1m']:
                        # 1-minute data: max 7 days
                        time_options = ["1 day", "2 days", "5 days", "7 days"]
                        time_map = {"1 day": "1d", "2 days": "2d", "5 days": "5d", "7 days": "7d"}
                        default_index = 2  # 5 days
                    elif interval in ['5m', '15m']:
                        # 5m/15m data: max 60 days
                        time_options = ["1 day", "5 days", "1 month", "2 months"]
                        time_map = {"1 day": "1d", "5 days": "5d", "1 month": "1mo", "2 months": "2mo"}
                        default_index = 2  # 1 month
                    elif interval in ['1h']:
                        # 1h data: max 2 years
                        time_options = ["1 day", "5 days", "1 month", "6 months", "1 year", "2 years"]
                        time_map = {"1 day": "1d", "5 days": "5d", "1 month": "1mo", "6 months": "6mo", "1 year": "1y", "2 years": "2y"}
                        default_index = 3  # 6 months
                    else:  # 1d
                        time_options = ["1 day", "5 days", "1 month", "6 months", "1 year", "3 years", "5 years", "all time"]
                        time_map = {"1 day": "1d", "5 days": "5d", "1 month": "1mo", "6 months": "6mo", "1 year": "1y", "3 years": "3y", "5 years": "5y", "all time": "max"}
                        default_index = 5 
                    
                    range_label = st.selectbox(
                        "Time Range:",
                        time_options,
                        index=default_index,
                        key="time_range"
                    )
                    period = time_map[range_label]
                    
                    if interval in ['1m']:
                        st.caption("1-minute data limited to 7 days max")
                    elif interval in ['5m', '15m']:
                        st.caption("5m/15m data limited to 60 days max")
                    elif interval in ['1h']:
                        st.caption("1-hour data limited to 2 years max")
                
                with st.expander("Moving Average Settings"):
                    st.markdown("#### Moving Average Settings")
                    ma_col1, ma_col2, ma_col3 = st.columns(3)
                    
                    if interval == '1d':
                        # Daily data: Use traditional day-based moving averages
                        short_default, short_min, short_max = 20, 5, 100
                        long_default, long_min, long_max = 50, 20, 300
                        ma_unit = "days"
                    elif interval == '1h':
                        # Hourly data: Use hour-based moving averages
                        short_default, short_min, short_max = 24, 6, 168  # 1 day to 1 week
                        long_default, long_min, long_max = 168, 24, 720   # 1 week to 1 month
                        ma_unit = "hours"
                    elif interval in ['5m', '15m']:
                        # 5/15-minute data: Use period-based moving averages
                        short_default, short_min, short_max = 20, 5, 100
                        long_default, long_min, long_max = 60, 20, 200
                        ma_unit = "periods"
                    else:  # 1m
                        # 1-minute data: Use smaller periods
                        short_default, short_min, short_max = 20, 5, 120
                        long_default, long_min, long_max = 60, 20, 240
                        ma_unit = "minutes"
                    
                    with ma_col1:
                        show_ma = st.checkbox("Show Moving Averages", value=True, key="show_moving_averages", help="20-day MA = average of today's price + previous 19 days.")
                    
                    with ma_col2:
                        short_ma = st.slider(
                            f"Short MA ({ma_unit})",
                            min_value=short_min,
                            max_value=short_max,
                            value=short_default,
                            step=1,
                            key="short_ma",
                            disabled=not show_ma
                        )
                    
                    with ma_col3:
                        long_ma = st.slider(
                            f"Long MA ({ma_unit})",
                            min_value=long_min,
                            max_value=long_max,
                            value=long_default,
                            step=1,
                            key="long_ma",
                            disabled=not show_ma
                        )
                    
                    if show_ma:
                        st.caption("Moving averages smooth out price action to identify trends. When short MA crosses above long MA, it may signal an uptrend.")
                    
                # Fetch data
                try:
                    data = yf.Ticker(selected_stock_data['ticker']).history(interval=interval, period=period)
                    original_data = data.copy() 
                    
                    if interval in ['1m', '5m', '15m', '1h']:
                        if isinstance(data.index, pd.DatetimeIndex):
                            if data.index.tz is None:
                                data = data.tz_localize('UTC').tz_convert('US/Eastern')
                            else:
                                data = data.tz_convert('US/Eastern')
                            data = data.between_time('09:30', '16:00')
                            data = data.tz_localize(None) 
                        else:
                            st.warning("Data index is not a DatetimeIndex. Skipping market hours filtering.")
                    
                except Exception as e:
                    data = None
                    original_data = None
                    error_msg = str(e)
                    
                    if "not available" in error_msg and "days" in error_msg:
                        st.error(f"**Data Limitation Exceeded**")
                        st.error(f"The selected combination of **{interval_label}** interval and **{range_label}** range exceeds Yahoo Finance limits.")
                        
                        if interval in ['1m']:
                            st.info("For 1-minute data, try a shorter time range (max 7 days)")
                        elif interval in ['5m', '15m']:
                            st.info("For 5/15-minute data, try a shorter time range (max ~60 days) or switch to daily data for longer periods")
                        elif interval in ['1h']:
                            st.info("For hourly data, try a shorter time range (max ~2 years) or switch to daily data")
                    else:
                        st.warning(f"Error fetching data: {e}")
                
                if data is not None and not data.empty:
                    short_ma_data = None
                    long_ma_data = None
                    
                    if show_ma and len(data) > max(short_ma, long_ma):
                        short_ma_data = data['Close'].rolling(window=short_ma, min_periods=1).mean()
                        long_ma_data = data['Close'].rolling(window=long_ma, min_periods=1).mean()
                    
                    if chart_type == "Line Chart":
                        fig_chart = go.Figure()
                        
                        fig_chart.add_trace(go.Scatter(
                            x=data.index,
                            y=data['Close'],
                            mode='lines',
                            line=dict(color='#fafafa', width=2),
                            name=selected_stock_data['ticker'],
                            hovertemplate='<b>%{fullData.name}</b><br>' +
                                        'Date: %{x}<br>' +
                                        'Price: $%{y:.2f}<extra></extra>'
                        ))
                        
                        if show_ma:
                            if short_ma_data is not None:
                                fig_chart.add_trace(go.Scatter(
                                    x=data.index,
                                    y=short_ma_data,
                                    mode='lines',
                                    line=dict(color='#ff6b6b', width=1.5),
                                    name=f'{short_ma}-{ma_unit} MA',
                                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                                'Date: %{x}<br>' +
                                                'MA: $%{y:.2f}<extra></extra>'
                                ))
                            
                            if long_ma_data is not None:
                                fig_chart.add_trace(go.Scatter(
                                    x=data.index,
                                    y=long_ma_data,
                                    mode='lines',
                                    line=dict(color='#4ecdc4', width=1.5),
                                    name=f'{long_ma}-{ma_unit} MA',
                                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                                'Date: %{x}<br>' +
                                                'MA: $%{y:.2f}<extra></extra>'
                                ))
                        
                        chart_title = f"{selected_stock_data['ticker']} Price Chart ({interval_label} interval, {range_label} range)"
                    else:  # Candlestick Chart
                        fig_chart = go.Figure()
                        
                        fig_chart.add_trace(go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            increasing_line_color='#26a69a',
                            decreasing_line_color='#ef5350',
                            increasing_fillcolor='#26a69a',
                            decreasing_fillcolor='#ef5350',
                            name=selected_stock_data['ticker']
                        ))
                        
                        if show_ma:
                            if short_ma_data is not None:
                                fig_chart.add_trace(go.Scatter(
                                    x=data.index,
                                    y=short_ma_data,
                                    mode='lines',
                                    line=dict(color='#ff6b6b', width=1.5),
                                    name=f'{short_ma}-{ma_unit} MA',
                                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                                'Date: %{x}<br>' +
                                                'MA: $%{y:.2f}<extra></extra>'
                                ))
                            
                            if long_ma_data is not None:
                                fig_chart.add_trace(go.Scatter(
                                    x=data.index,
                                    y=long_ma_data,
                                    mode='lines',
                                    line=dict(color='#4ecdc4', width=1.5),
                                    name=f'{long_ma}-{ma_unit} MA',
                                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                                'Date: %{x}<br>' +
                                                'MA: $%{y:.2f}<extra></extra>'
                                ))
                        
                        chart_title = f"{selected_stock_data['ticker']} Candlestick Chart ({interval_label})"
                    
                    layout_updates = {
                        "title": chart_title,
                        "xaxis_title": "Time",
                        "yaxis_title": "Price ($)",
                        "height": 500,
                        "xaxis_rangeslider_visible": False,
                        "template": "plotly_dark",
                        "title_font_size": 18,
                        "font": dict(size=16),
                        "xaxis": dict(title_font_size=16),
                        "yaxis": dict(title_font_size=16),
                        "legend": dict(font=dict(size=16)),
                        "hoverlabel": dict(font_size=16)
                    }
                    
                    if original_data is not None and not original_data.empty:
                        x_range = [original_data.index.min(), original_data.index.max()]
                        layout_updates["xaxis"] = {
                            "range": x_range,
                            "type": "date"
                        }
                        
                        if interval in ['1m', '5m', '15m', '1h']:
                            layout_updates["xaxis"]["rangebreaks"] = [
                                {"pattern": "hour", "bounds": [16, 9.5]},
                                {"pattern": "day of week", "bounds": [6, 1]},
                                {"values": ["2023-12-25", "2024-01-01"]}
                            ]
                        elif interval == '1d':
                            layout_updates["xaxis"]["rangebreaks"] = [
                                {"pattern": "day of week", "bounds": [6, 1]},
                                {"values": ["2023-12-25", "2024-01-01"]}
                            ]
                    
                    fig_chart.update_layout(**layout_updates)
                    st.plotly_chart(fig_chart, use_container_width=True)

                
                period_change_pct = None
                period_change_color = "gray"
                if data is not None and not data.empty and len(data) > 1:
                    first_price = data['Close'].iloc[0]
                    last_price = data['Close'].iloc[-1]
                    period_change_pct = ((last_price - first_price) / first_price) * 100
                    period_change_color = "green" if period_change_pct >= 0 else "red"
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if period_change_pct is not None:
                        price_str = f"${selected_stock_data['last_price']:.2f}" if selected_stock_data['last_price'] is not None else "N/A"
                        st.metric(
                            "Current Price", 
                            price_str,
                            delta=f"{period_change_pct:+.1f}% ({range_label})"
                        )
                    else:
                        price_str = f"${selected_stock_data['last_price']:.2f}" if selected_stock_data['last_price'] is not None else "N/A"
                        st.metric("Current Price", price_str)
                with col2:
                    day_high_str = f"${selected_stock_data['day_high']:.2f}" if selected_stock_data['day_high'] is not None else "N/A"
                    st.metric("Day High", day_high_str)
                with col3:
                    day_low_str = f"${selected_stock_data['day_low']:.2f}" if selected_stock_data['day_low'] is not None else "N/A"
                    st.metric("Day Low", day_low_str)
                with col4:
                    volume_display = f"{selected_stock_data['volume']:,}" if selected_stock_data['volume'] is not None else "N/A"
                    st.metric("Volume", volume_display)
                
                # st.markdown("### Summary Statistics")
                # col1, col2 = st.columns(2)
                # with col1:
                #     bid_str = f"${selected_stock_data['bid']:.2f}" if selected_stock_data['bid'] is not None else "N/A"
                #     ask_str = f"${selected_stock_data['ask']:.2f}" if selected_stock_data['ask'] is not None else "N/A"
                #     st.write(f"**Bid/Ask:** {bid_str} / {ask_str}")
                    
                #     year_high_str = f"${selected_stock_data['year_high']:.2f}" if selected_stock_data['year_high'] is not None else "N/A"
                #     st.write(f"**52W High:** {year_high_str}")
                    
                #     year_low_str = f"${selected_stock_data['year_low']:.2f}" if selected_stock_data['year_low'] is not None else "N/A"
                #     st.write(f"**52W Low:** {year_low_str}")
                # with col2:
                #     if selected_stock_data['market_cap'] is not None:
                #         market_cap_bil = selected_stock_data['market_cap'] / 1e9
                #         st.write(f"**Market Cap:** {market_cap_bil:,.2f} B")
                #     else:
                #         st.write(f"**Market Cap:** N/A")
                    
                #     change_str = f"{selected_stock_data['change']:+.2f}" if selected_stock_data['change'] is not None else "N/A"
                #     change_pct_str = f"{selected_stock_data['change_pct']:+.2f}%" if selected_stock_data['change_pct'] is not None else "N/A"
                #     st.write(f"**Change:** {change_str} ({change_pct_str})")
                
                with st.expander("Risk Metrics & Analysis"): 
                    if data is not None and len(data) > 1:  
                        st.markdown("### Risk Metrics & Analysis")
                        
                        returns = data['Close'].pct_change().dropna()
                        
                        if len(returns) > 1:
                            if interval == '1d':
                                annualization_factor = 252  # Trading days per year
                                period_label = "Daily"
                            elif interval == '1h':
                                annualization_factor = 252 * 24  # Hours per year (trading days)
                                period_label = "Hourly"
                            elif interval == '15m':
                                annualization_factor = 252 * 24 * 4  # 15-min periods per year
                                period_label = "15-min"
                            elif interval == '5m':
                                annualization_factor = 252 * 24 * 12  # 5-min periods per year
                                period_label = "5-min"
                            elif interval == '1m':
                                annualization_factor = 252 * 24 * 60  # 1-min periods per year
                                period_label = "1-min"
                            else:
                                annualization_factor = 252
                                period_label = "Daily"
                            
                            avg_return = returns.mean()
                            volatility = returns.std()
                            
                            annualized_return = avg_return * annualization_factor
                            annualized_volatility = volatility * (annualization_factor ** 0.5)
                            
                            risk_free_rate = 0.02
                            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
                            
                            risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
                            
                            with risk_col1:
                                st.metric(
                                    "Annualized Return",
                                    f"{annualized_return:.2%}",
                                    delta=None,
                                )
                            
                            with risk_col2:
                                st.metric(
                                    "Annualized Volatility",
                                    f"{annualized_volatility:.2%}",
                                    delta=None,
                                    help = "Annualized volatility is the standard deviation of returns."
                                )
                            
                            with risk_col3:
                                st.metric(
                                    "Sharpe Ratio",
                                    f"{sharpe_ratio:.2f}",
                                    delta="Good (above 1)" if sharpe_ratio > 1 else "Poor" if sharpe_ratio < 0 else "Fair",
                                    help="Sharpe Ratio is calculated as (Annualized Return - Risk-Free Rate) / Annualized Volatility. A higher Sharpe Ratio indicates better risk-adjusted returns."
                                )
                            
                            with risk_col4:
                                var_95 = returns.quantile(0.05)  
                                st.metric(
                                    "VaR (95%)",
                                    f"{var_95:.2%}",
                                    delta="Max 1-day loss",
                                    help="Value at Risk at 95% confidence level indicates the maximum expected loss that will NOT be exceeded on 95% of days. There's a 5% chance that daily losses could be worse than this value."
                                )
                            
                            st.markdown("#### Interactive Value-at-Risk Analysis")
                            var_col1, var_col2 = st.columns([1, 2])
                            
                            with var_col1:
                                confidence_level = st.slider(
                                    "Confidence Level (%)",
                                    min_value=90,
                                    max_value=99,
                                    value=95,
                                    step=1,
                                    key="var_confidence"
                                )
                                
                                var_percentile = (100 - confidence_level) / 100
                                var_value = returns.quantile(var_percentile)
                                
                                if selected_stock_data['last_price'] is not None:
                                    delta_str = f"${var_value * selected_stock_data['last_price']:.2f} loss (Max loss per share)"
                                else:
                                    delta_str = "Max loss per share (N/A)"
                                
                                st.metric(
                                    f"VaR ({confidence_level}%)",
                                    f"{var_value:.2%}",
                                    delta=delta_str
                                )
                                
                                st.caption(f"With {confidence_level}% confidence, losses won't exceed {abs(var_value):.2%} in one {period_label.lower()} period")
                            
                            with var_col2:
                                confidence_levels = list(range(90, 100))
                                var_values = [returns.quantile((100 - cl) / 100) for cl in confidence_levels]
                                
                                fig_var = go.Figure()
                                fig_var.add_trace(go.Scatter(
                                    x=confidence_levels,
                                    y=[abs(v) for v in var_values],  
                                    mode='lines+markers',
                                    line=dict(color='#ff6b6b', width=2),
                                    marker=dict(size=6),
                                    name='VaR Curve',
                                    hovertemplate='<b>Confidence Level: %{x}%</b><br>' +
                                                'VaR: %{y:.2%}<extra></extra>'
                                ))
                                
                                fig_var.add_trace(go.Scatter(
                                    x=[confidence_level],
                                    y=[abs(var_value)],
                                    mode='markers',
                                    marker=dict(size=12, color='#4ecdc4', symbol='diamond'),
                                    name='Selected Level',
                                    hovertemplate=f'<b>Selected: {confidence_level}%</b><br>' +
                                                f'VaR: {abs(var_value):.2%}<extra></extra>'
                                ))
                                
                                fig_var.update_layout(
                                    title=f"VaR Curve - {selected_stock_data['ticker']} ({period_label})",
                                    xaxis_title="Confidence Level (%)",
                                    yaxis_title="Value at Risk (absolute %)",
                                    height=300,
                                    template="plotly_dark",
                                    showlegend=False,
                                    title_font_size=18,
                                    font=dict(size=16),
                                    xaxis=dict(title_font_size=16),
                                    yaxis=dict(title_font_size=16),
                                    hoverlabel=dict(font_size=16)
                                )
                                
                                st.plotly_chart(fig_var, use_container_width=True)
                        else:
                            st.warning("Insufficient data to calculate risk metrics.")
                    else:
                        st.info("Risk metrics require at least 2 data points.")
                col1, col2 = st.columns(2)
                with col1:
                    if data is not None and not data.empty:
                        df_export = data.reset_index()
                        if 'Datetime' in df_export.columns:
                            df_export = df_export[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
                        elif data.index.name:
                            df_export = df_export[[data.index.name, 'Open', 'High', 'Low', 'Close', 'Volume']]
                        else:
                            df_export = df_export[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                        
                        st.download_button(
                            label="Download CSV",
                            data=df_export.to_csv(index=False),
                            file_name=f"{selected_stock_data['ticker']}_{interval.replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
                with col2:
                    st.write("")
                st.markdown(f"[View {selected_stock_data['ticker']} on Yahoo Finance](https://finance.yahoo.com/quote/{selected_stock_data['ticker']})")
                st.markdown('</div>', unsafe_allow_html=True)
