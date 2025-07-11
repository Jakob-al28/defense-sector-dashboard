import streamlit as st
import warnings
from utils.news_analysis import news_analysis
warnings.filterwarnings('ignore')


def render_tab3():
    st.title("News Analysis")
    st.write("Many sources state that the best way to predict the market is to gather as much data as possible. For that NLP is a great tool. \n That's why in this section we'll look at news sentiment toward the defense industry and the stock market, \n using Rheinmetall as our example.")
    st.write("---")
    st.write("The news were fetched from the NewsAPI and analyzed using VADER sentiment analysis. The results are shown below.")
    news_analysis()
    st.write(
        "It can be seen that the sentiment towards Rheinmetall is slightly more positive, with headlines underlining the recent new contracts for Rheinmetall and indicating a positive outlook. "
        "Further analysis could also examine social media sentiment (Reddit, X (Twitter) etc.) to get a broader view. "   
        "While predicting the market is difficult, many factors, such as NATO’s decision to increase defense spending and other market indicators, suggest an upward trajectory. \n Therefore, over the the next years, the defense industry is likely to grow, even if a short-term correction may occur."     
    )
    st.markdown("---")
    st.markdown("### Summary")
    st.write("This dashboard uncovers three insights that underpin an optimistic outlook for the defense industry:")
    st.write("In the first section, the 'Overview' tab, data shows that a majority of NATO member states are currently falling short of the alliance’s 5-percent defense spending guideline. This shortfall points to a pipeline of planned investments over the coming years as governments work to meet their commitments.")
    st.write("Second, the 'Markets' segment highlights a steady upward trajectory in defense sector valuations. Year-on-year comparisons demonstrate that defense equities and related indices have maintained growth, reflecting strong investor confidence in the industry’s resilience. Although some firms, such as Rheinmetall, demonstrate higher volatility, which could signal a near-term correction, these fluctuations are unlikely to derail the sector’s long-term expansion prospects.")
    st.write("Third, the 'News Analysis' section captures prevailing positive sentiment across media outlets. Coverage of recent major contracts, most notably those secured by Rheinmetall, has trended favorably, suggesting broad market approval of new deal announcements.")
    st.write("Taken together, the requirement for increased NATO defense outlays, the solid market performance of defense companies, and the upbeat tone of industry news all point toward sustained long-term expansion in the defense sector.")