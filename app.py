# run this streamlit app: streamlit run app.py

import streamlit as st
from tabs.tab1 import render_tab1
import streamlit as st
from tabs.tab2 import render_tab2
from tabs.tab3 import render_tab3

st.set_page_config(
    page_title="An outlook on global affairs",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp {
        max-width: 1200px !important;
        margin: auto;
    }
    
    .main .block-container {
        max-width: 1200px !important;
        padding-left: 5rem !important;
        padding-right: 5rem !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        width: 100% !important;
        max-width: 1200px !important;
    }
    
    .stTabs [role="tabpanel"] {
        width: 100% !important;
        max-width: 1200px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("An outlook on global affairs - Interactive Dashboard")

tab1, tab2, tab3 = st.tabs(["Overview", "Markets", "News Analysis"])

with tab1:
    render_tab1()

with tab2:
    render_tab2()

with tab3:
    render_tab3()
