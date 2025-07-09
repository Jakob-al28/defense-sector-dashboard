import streamlit as st
from streamlit.components.v1 import html
from utils.stocks import render_stocks
from utils.compare import render_compare

def switch(tab_idx: int) -> str:
    return f"""
        <script>
        var tabGroup = window.parent.document.getElementsByClassName("stTabs")[0];
        var tabButtons = tabGroup.getElementsByTagName("button");
        tabButtons[{tab_idx}].click();
        </script>
        """

def render_tab2():
    st.write("The defense industry is a complex and multifaceted sector. \n It includes a wide range of products and services, from weapons systems and vehicles to communication and surveillance equipment. \n The industry is heavily influenced by government policies, international relations, and technological advancements.")
    st.write("---")
    st.write("To get a better financial understanding of the defense industry, we will take a look at the stock market. \n Below, there is a table that shows the stock prices of some defense companies and ETFs. \n We will take a closer look at Rheinmetall's financial performance. You may also click on a row to display the stock price.")
    render_stocks()
    st.write(
        "Rheinmetall has seen a significant increase in its stock price since the start of the Ukraine conflict. "
        "Over the past six months, the stock has risen by approximately 190%.\n\n"
        "If we look at the SPDR S&P Aerospace & Defense ETF (PPA), it has gained about 23% over the last six months "
        "and roughly 100% since the beginning of the Ukraine conflict."
    )
    st.write("Further metrics like the moving average and the risk analysis also shows the performance of the sector.")
    st.write(
        "As this growth influences industry activity, several counterarguments arise. "
        "Sustainability is a major concern, since the defense industry is often associated with environmental degradation and human rights violations. Some argue, however, that the defense industry is necessary for national security and that it can contribute to economic growth and job creation, and thus be considered sustainable."
    )
    st.write(
        "We will now take a closer look at sustainability by comparing ESG funds with defense funds, and then contrasting ESG funds that regard defense activities as sustainable versus those that do not." 
    )
    st.write("---")
    render_compare()
    st.write("---")
    st.write(
        "The analysis shows that the defense industry has been outperforming the ESG sector over the past six months.\n"
        "Taking NATO’s goals into account, there will be changes. We will now take a look at the news sentiment towards this change."
    )
    if st.button("Proceed to News Analysis.", use_container_width=True, key="tab3_button"):
        html(switch(2), height=0)