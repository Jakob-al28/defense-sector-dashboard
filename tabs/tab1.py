import streamlit as st
from utils.maps import render_map
from utils.time_series import render_time_series
from utils.bar_chart import render_racing_bar_chart
from streamlit.components.v1 import html

def switch(tab_idx: int) -> str:
    return f"""
        <script>
        var tabGroup = window.parent.document.getElementsByClassName("stTabs")[0];
        var tabButtons = tabGroup.getElementsByTagName("button");
        tabButtons[{tab_idx}].click();
        </script>
        """

def render_tab1():
    st.markdown("## Overview")
    st.markdown("Recent shifts in geopolitical tensions have led to a increase in defense spending. \n NATO has set a 5% GDP target for defense spending by 2027. \n The economy has not been growing as expected lately, and sustainability is becoming a more important factor.")
    st.markdown("In this dashboard, we will take a look at these areas and analyze them.")
    st.markdown("---")
    st.markdown("Numerous conflicts have caused NATO to make the decision to increase defense spending. The Russo-Ukrainian war, the war in Syria, and the war in Afghanistan have all contributed to this decision.")
    st.markdown("Let's assume that all NATO countries will spend 5% of their GDP on defense by 2027. \n This is a significant increase for some NATO countries. \n The map below shows the current military spending of all european countries.")
    data_type = render_map()
    st.markdown("Europe shows a clear eastward trend in military spending. Western European states have lower military spending at around 2-4% of government spending, while the trend rises continuously toward the north and east, reaching 8-9% in the Baltic states and Poland. Ukraine, Belarus, and Russia are at the top of the list.")    
    st.markdown("To take the temporal dimension into perspective, we can look at the change in military spending over the last years. The time series below shows this change.")
    st.markdown("---")
    render_time_series(data_type, clicked_countries=st.session_state.get("clicked_countries", []))
    st.markdown(
        "If you’d like to see how these indicators have evolved over time, the racing bar chart below compares NATO members with other countries."
    )
    with st.expander("Racing Bar Chart"):
        render_racing_bar_chart(data_type)
    st.markdown("Historically, many countries reduced their military expenditure. However, this trend has recently reversed, with numerous nations now increasing their defense spending. \n To meet NATO's 5% GDP target, countries like Germany and France face substantial increases in military budgets compared to other NATO members. \n These changes, particularly for richer countries (see GDP World Map above), will likely impact the defense industry.")    
    if st.button("Find out about the changes in the Defense Industry.", use_container_width=True, key="tab2_button"):
        html(switch(1), height=0)


    st.markdown("---")
    st.markdown("Data Sources:")
    st.markdown("[Global Peace Index](https://fragilestatesindex.org/global-data/)")
    st.markdown("[Global CO2 Emissions](https://github.com/owid/co2-data/blob/master/owid-co2-data.csv)")
    st.markdown("[Global Military Expenditure](https://sipri.org/databases/milex)")
    st.markdown("[Global GDP](https://data.worldbank.org/indicator/NY.GDP.MKTP.CD)")
    st.markdown("[Global GDP Change](https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG)")