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