import streamlit as st
import pandas as pd
import yfinance as yf
from backend import data_fetching
import plotly.graph_objects as go

def render_compare():
    st.markdown("### Defense vs Vanguard FTSE All-World: A Performance Comparison")
    
    # Define sector groups
    defense_tickers = ["RTX", "PPA", "RHM.F", "HON", "LMT", "XAR", "ITA"]
    # esg_tickers = ["VSGX", "DSI", "EAGG", "ESGU", "ESGD", "ESGE", "CHGX"]
    esg_tickers = ["SPY", "IVV", "VOO", "VTI", "QQQ", "IWM", "DIA", "EEM", "GLD", "AGG"]
    # Fetch data for both sectors
    with st.spinner("Loading sector data..."):
        defense_data = data_fetching.get_top_stocks_quotes(defense_tickers)
        esg_data = data_fetching.get_top_stocks_quotes(esg_tickers)
    
    if not isinstance(defense_data, pd.DataFrame) or not isinstance(esg_data, pd.DataFrame):
        st.error("Error loading sector data")
        return
    
    # Calculate sector averages
    defense_avg_change = defense_data['change_pct'].mean()
    esg_avg_change = esg_data['change_pct'].mean()
    
    # Display sector comparison metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Defense Sector", 
            f"{defense_avg_change:+.1f}%",
            delta="Daily Average",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "All-World ETF", 
            f"{esg_avg_change:+.1f}%",
            delta="Daily Average", 
            delta_color="normal"
        )
    
    with col3:
        performance_gap = defense_avg_change - esg_avg_change
        st.metric(
            "Performance Gap",
            f"{performance_gap:+.1f}%",
            delta="Defense vs Global Market",
            delta_color="normal"
        )
        
    # Get historical data for representative ETFs
    defense_etf = "PPA"  # SPDR S&P Aerospace & Defense ETF
    esg_etf = "VGWL.DE"     # iShares MSCI KLD 400 Social ETF
    
    try:
        defense_hist = yf.Ticker(defense_etf).history(period="6mo")
        esg_hist = yf.Ticker(esg_etf).history(period="6mo")
        
        if not defense_hist.empty and not esg_hist.empty:
            # Calculate normalized performance (starting from 100)
            defense_norm = (defense_hist['Close'] / defense_hist['Close'].iloc[0]) * 100
            esg_norm = (esg_hist['Close'] / esg_hist['Close'].iloc[0]) * 100
            
            # Create comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=defense_norm.index,
                y=defense_norm.values,
                mode='lines',
                name='Defense (PPA)',
                line=dict(color='#ff6b6b', width=3),
                hovertemplate='<b>Defense Sector</b><br>' +
                            'Date: %{x}<br>' +
                            'Performance: %{y:.1f}%<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=esg_norm.index,
                y=esg_norm.values,
                mode='lines',
                name='Vanguard FTSE All-World (VGWL.DE)',
                line=dict(color='#4ecdc4', width=3),
                hovertemplate='<b>Global Market ETF</b><br>' +
                            'Date: %{x}<br>' +
                            'Performance: %{y:.1f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title="6-Month Normalized Performance (Base: 100)",
                xaxis_title="Date",
                yaxis_title="Normalized Performance",
                height=400,
                template="plotly_dark",
                hovermode='x unified',
                title_font_size=18,
                font=dict(size=16),
                xaxis=dict(title_font_size=16),
                yaxis=dict(title_font_size=16),
                legend=dict(font=dict(size=16)),
                hoverlabel=dict(font_size=16)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display performance metrics
            defense_total_return = (defense_norm.iloc[-1] - 100)
            esg_total_return = (esg_norm.iloc[-1] - 100)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Defense 6M Return", f"{defense_total_return:+.1f}%")
            with col2:
                st.metric("Global Market ETF 6M Return", f"{esg_total_return:+.1f}%")
                
    except Exception as e:
        st.warning("Unable to load historical performance data")
    
    # Sector composition and key insights
    # st.markdown("### Key Insights")
    
    # col1, col2 = st.columns(2)
    
    # with col1:
    #     st.markdown("#### Defense Sector Highlights")
    #     st.write("• **Key Players**: Raytheon, Lockheed Martin, Rheinmetall")
        
    #     # Show top performers
    #     if not defense_data.empty:
    #         top_defense = defense_data.nlargest(3, 'change_pct')[['ticker', 'name', 'change_pct']]
    #         st.markdown("**Top Performers Today:**")
    #         for _, row in top_defense.iterrows():
    #             st.write(f"• {row['ticker']}: {row['change_pct']:+.1f}%")
    
    # with col2:
    #     st.markdown("#### ESG Sector Highlights")
    #     st.write("• **Focus**: Environmental, Social, Governance criteria")
        
    #     # Show top performers
    #     if not esg_data.empty:
    #         top_esg = esg_data.nlargest(3, 'change_pct')[['ticker', 'name', 'change_pct']]
    #         st.markdown("**Top Performers Today:**")
    #         for _, row in top_esg.iterrows():
    #             st.write(f"• {row['ticker']}: {row['change_pct']:+.1f}%")
    
    # Risk comparison
    if len(defense_data) > 0 and len(esg_data) > 0:
        st.markdown("### Risk Profile Comparison")
        
        defense_volatility = defense_data['change_pct'].std()
        esg_volatility = esg_data['change_pct'].std()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Defense Volatility", f"{defense_volatility:.1f}%", delta="Daily volatility")
        with col2:
            st.metric("Global ETF Volatility", f"{esg_volatility:.1f}%", delta="Daily volatility")
        
        if defense_volatility > esg_volatility:
            st.write("Defense sector shows higher volatility, reflecting the impact of geopolitical events on defense stocks.")
        else:
            st.write("Global Market ETF shows higher volatility, possibly due to diverse holdings and market sentiment shifts.")