import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import requests

import os
import json
import glob
from dotenv import load_dotenv
from datetime import datetime, timedelta
from wordcloud import WordCloud, STOPWORDS
from lingua import Language, LanguageDetectorBuilder
from googletrans import Translator
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

def run_nlp(search_query="Rheinmetall"):
    load_dotenv()
    
    safe_query = "".join(c for c in search_query if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_query = safe_query.replace(' ', '_').lower()
    
    cache_pattern = f"data/news_cache/{safe_query}_news_*.json"
    cached_files = glob.glob(cache_pattern)
    
    latest_cache = None
    cache_age_days = None
    
    if cached_files:
        latest_cache = max(cached_files, key=os.path.getmtime)
        
        try:
            filename = os.path.basename(latest_cache)
            timestamp_str = filename.replace(f'{safe_query}_news_', '').replace('.json', '')
            cache_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            cache_age_days = (datetime.now() - cache_date).days
            
            print(f"Found cached file: {latest_cache}")
            print(f"Cache age: {cache_age_days} days")
            
        except ValueError:
            cache_age_days = 999 
    
    if latest_cache and cache_age_days is not None and cache_age_days < 7:
        try:
            with open(latest_cache, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                return cached_data
        except Exception as e:
            print(f"Error reading cache: {e}")
    
    print("Cache expired or missing - fetching new data...")
    
    API_KEY = os.getenv('API_KEY')
    
    if not API_KEY:
        print("ERROR: API_KEY not found in .env file")
        return None
    
    from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    url = f"https://newsapi.org/v2/everything?q={search_query}&from={from_date}&sortBy=publishedAt&apiKey={API_KEY}"
    
    try:
        response = requests.get(url)
        response_data = response.json()

        os.makedirs('data', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_filename = f"data/news_cache/{safe_query}_news_{timestamp}.json"
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)
        print(f"Saved new cache to: {json_filename}")
        
        if 'articles' in response_data and response_data['articles']:
            articles_df = pd.DataFrame(response_data['articles'])
            csv_filename = f"data/news_cache/{safe_query}_news_{timestamp}.csv"
            articles_df.to_csv(csv_filename, index=False, encoding='utf-8')
            print(f"Saved CSV to: {csv_filename}")
        
        return response_data
        
    except Exception as e:
        print(f"ERROR fetching news: {e}")
        if latest_cache:
            print("Using old cache as fallback...")
            try:
                with open(latest_cache, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return None

def display_nlp(news_data, search_query="Rheinmetall"):
    if not news_data or 'articles' not in news_data:
        st.error("No news data available for analysis")
        return
    
    articles = news_data['articles']
    if not articles:
        st.error("No articles found in news data")
        return
    

    
    def detect_and_translate(text):
        try:

            try:
                detector = LanguageDetectorBuilder.from_all_languages().build()
                translator = Translator()
            except Exception as e:
                st.error(f"Error initializing language detection: {e}")
                return
            detected_language = detector.detect_language_of(text)
            
            if detected_language is None:
                try:
                    translated = translator.translate(text, dest='en')
                    detected_code = translated.src if hasattr(translated, 'src') else 'unknown'
                    if hasattr(translated, 'text'):
                        return translated.text, detected_code
                    else:
                        return text, detected_code  
                except Exception as e:
                    print(e)
                    return text, 'unknown'
            
            lang_code_map = {
                Language.ENGLISH: 'en',
                Language.GERMAN: 'de', 
                Language.FRENCH: 'fr',
                Language.SPANISH: 'es',
                Language.ITALIAN: 'it',
                Language.DUTCH: 'nl',
                Language.PORTUGUESE: 'pt',
                Language.DANISH: 'da',
                Language.SWEDISH: 'sv',
                Language.FINNISH: 'fi',
                Language.RUSSIAN: 'ru',
                Language.POLISH: 'pl',
                Language.CZECH: 'cs',
                Language.HUNGARIAN: 'hu',
                Language.ROMANIAN: 'ro',
                Language.BULGARIAN: 'bg',
                Language.CROATIAN: 'hr',
                Language.SLOVAK: 'sk',
                Language.SLOVENE: 'sl',
                Language.LITHUANIAN: 'lt',
                Language.LATVIAN: 'lv',
                Language.ESTONIAN: 'et'
            }
            
            detected_code = lang_code_map.get(detected_language, 'unknown')
            
            if detected_code != 'en' and detected_code != 'unknown':
                try:
                    translated = translator.translate(text, src=detected_code, dest='en')
                    if hasattr(translated, 'text'):
                        return translated.text, detected_code
                    else:
                        return text, detected_code  
                except Exception as e:
                    print(e)
                    return text, detected_code
            else:
                return text, detected_code
        except Exception:
            return text, 'unknown'
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    headlines = []
    descriptions = []
    translated_headlines = []
    languages_detected = []
    
    st.markdown(f"### {search_query} News Analysis")
    st.markdown(f"**Analyzing {len(articles)} articles about '{search_query}' from the last 30 days**")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, article in enumerate(articles):
        progress_bar.progress((i + 1) / len(articles))
        status_text.text(f'Processing article {i + 1}/{len(articles)}...')
        
        if article.get('title'):
            original_title = article['title']
            headlines.append(original_title)
            
            translated_title, lang = detect_and_translate(original_title)
            translated_headlines.append(translated_title)
            languages_detected.append(lang)
        
        if article.get('description'):
            descriptions.append(article['description'])
    
    progress_bar.empty()
    status_text.empty()
    

    all_translated_text = translated_headlines + descriptions
    text_data = ' '.join([clean_text(text) for text in all_translated_text if text])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Most Frequent Words")
        
        enhanced_stopwords = set(STOPWORDS)
        enhanced_stopwords.update([
            # Company names and business terms
            'rheinmetall', 'ag', 'gmbh', 'inc', 'ltd', 'corp', 'company', 'group', 'holdings',
            'corporation', 'limited', 'plc', 'sa', 'spa', 'bv', 'nv', 'oy', 'ab', "w", "det", "pn", "fr", "az", "im"
            
            # News and media terms
            'said', 'says', 'will', 'also', 'new', 'one', 'two', 'would', 'could', 'should',
            'according', 'reuters', 'reported', 'report', 'news', 'article', 'story', 'breaking',
            'update', 'sources', 'source', 'press', 'release', 'statement', 'announced', 'announce',
            'tells', 'told', 'reports', 'reporting', 'correspondent', 'editor', 'journalist',
            
            # English common words
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one',
            'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'old', 'see', 'two',
            'who', 'boy', 'did', 'now', 'way', 'come', 'came', 'gone', 'went', 'been', 'have', 'than',
            'that', 'this', 'they', 'them', 'their', 'there', 'where', 'when', 'what', 'which', 'while',
            'with', 'from', 'into', 'over', 'under', 'after', 'before', 'through', 'between', 'among',
            'make', 'made', 'take', 'taken', 'give', 'given', 'know', 'known', 'think', 'thought',
            'look', 'looked', 'find', 'found', 'work', 'worked', 'call', 'called', 'try', 'tried',
            'need', 'needed', 'want', 'wanted', 'use', 'used', 'help', 'helped', 'tell', 'told',
            'ask', 'asked', 'feel', 'felt', 'seem', 'seemed', 'leave', 'left', 'move', 'moved',
            
            # German common words (articles, pronouns, prepositions, etc.)
            'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einen', 'einem', 'einer', 'eines',
            'und', 'oder', 'aber', 'doch', 'dann', 'wenn', 'weil', 'dass', 'als', 'wie', 'wo', 'was',
            'wer', 'wen', 'wem', 'wessen', 'welche', 'welcher', 'welches', 'dieser', 'diese', 'dieses',
            'jener', 'jene', 'jenes', 'alle', 'jeder', 'jede', 'jedes', 'manche', 'mancher', 'manches',
            'mit', 'von', 'aus', 'bei', 'nach', 'vor', 'über', 'unter', 'durch', 'für', 'ohne', 'gegen',
            'auf', 'an', 'in', 'zu', 'zwischen', 'seit', 'bis', 'während', 'wegen', 'trotz', 'statt',
            'ist', 'sind', 'war', 'waren', 'bin', 'bist', 'hat', 'haben', 'hatte', 'hatten', 'wird',
            'werden', 'wurde', 'wurden', 'kann', 'können', 'konnte', 'konnten', 'muss', 'müssen',
            'musste', 'mussten', 'soll', 'sollen', 'sollte', 'sollten', 'will', 'wollen', 'wollte',
            'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr', 'mich', 'dich', 'sich', 'uns', 'euch',
            'mir', 'dir', 'ihm', 'ihr', 'ihnen', 'mein', 'dein', 'sein', 'unser', 'euer',
            'nicht', 'kein', 'keine', 'keinen', 'keinem', 'keiner', 'keines', 'nur', 'auch', 'noch',
            'schon', 'bereits', 'immer', 'nie', 'niemals', 'oft', 'manchmal', 'selten', 'heute',
            'gestern', 'morgen', 'hier', 'dort', 'da', 'sehr', 'ganz', 'ziemlich', 'etwas', 'nichts',
            
            # French common words
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'mais', 'car', 'donc',
            'que', 'qui', 'quoi', 'où', 'quand', 'comment', 'pourquoi', 'ce', 'cette', 'ces', 'cet',
            'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre', 'nos', 'votre', 'vos',
            'leur', 'leurs', 'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'me', 'te',
            'se', 'moi', 'toi', 'lui', 'eux', 'dans', 'sur', 'avec', 'sans', 'pour', 'par', 'vers',
            'chez', 'sous', 'entre', 'depuis', 'pendant', 'avant', 'après', 'contre', 'selon',
            'est', 'sont', 'était', 'étaient', 'été', 'être', 'avoir', 'avait', 'avaient',
            'aura', 'auront', 'fait', 'faire', 'fais', 'faisait', 'faisaient', 'fera', 'feront',
            'peut', 'peuvent', 'pouvait', 'pouvaient', 'pourra', 'pourront', 'doit', 'doivent',
            'pas', 'plus', 'moins', 'très', 'assez', 'trop', 'bien', 'mal', 'beaucoup', 'peu',
            'tout', 'tous', 'toute', 'toutes', 'rien', 'quelque', 'chose', 'aucun', 'aucune',
            'ici', 'là', 'maintenant', 'hier', 'demain', 'aujourd', 'hui', 'jamais', 'toujours',
            
            # Spanish common words
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'del', 'de', 'y', 'o', 'pero',
            'que', 'quien', 'quienes', 'qué', 'cuál', 'cuáles', 'donde', 'cuándo', 'cómo', 'por',
            'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas', 'aquel', 'aquella',
            'mi', 'mis', 'tu', 'tus', 'su', 'sus', 'nuestro', 'nuestra', 'yo', 'tú', 'él',
            'ella', 'nosotros', 'nosotras', 'ellos', 'ellas', 'me', 'te', 'se', 'nos', 'os',
            'en', 'con', 'sin', 'para', 'por', 'desde', 'hasta', 'durante', 'antes', 'después',
            'es', 'son', 'era', 'eran', 'fue', 'fueron', 'será', 'serán', 'ser', 'estar',
            'haber', 'he', 'has', 'ha', 'hemos', 'han', 'había', 'hacer', 'hago', 'hace',
            'poder', 'puedo', 'puede', 'pueden', 'podía', 'no', 'sí', 'más', 'menos', 'muy',
            'mucho', 'poco', 'todo', 'nada', 'algo', 'aquí', 'ahí', 'ahora', 'ayer', 'hoy',
            
            # Italian common words
            'il', 'lo', 'la', 'gli', 'le', 'un', 'uno', 'una', 'del', 'dello', 'della', 'dei',
            'degli', 'delle', 'al', 'allo', 'alla', 'agli', 'alle', 'dal', 'dallo', 'dalla',
            'dai', 'dagli', 'dalle', 'nel', 'nello', 'nella', 'nei', 'negli', 'nelle', 'sul',
            'sullo', 'sulla', 'sui', 'sugli', 'sulle', 'di', 'a', 'da', 'in', 'con', 'su',
            'per', 'tra', 'fra', 'e', 'o', 'ma', 'però', 'anche', 'ancora', 'già', 'non',
            'che', 'chi', 'cosa', 'come', 'quando', 'dove', 'perché', 'questo', 'questa',
            'questi', 'queste', 'quello', 'quella', 'quelli', 'quelle', 'io', 'tu', 'lui',
            'lei', 'noi', 'voi', 'loro', 'mi', 'ti', 'si', 'ci', 'vi', 'li', 'le', 'ne',
            'è', 'sono', 'era', 'erano', 'essere', 'avere', 'ho', 'hai', 'ha', 'abbiamo',
            'avete', 'hanno', 'aveva', 'avevano', 'fare', 'faccio', 'fai', 'fa', 'facciamo',
            'fate', 'fanno', 'potere', 'posso', 'puoi', 'può', 'possiamo', 'potete', 'possono',
            
            # Dutch common words
            'de', 'het', 'een', 'van', 'en', 'in', 'op', 'met', 'voor', 'aan', 'bij', 'naar',
            'over', 'door', 'uit', 'onder', 'tegen', 'vanaf', 'binnen', 'buiten', 'zonder',
            'tijdens', 'na', 'om', 'rond', 'langs', 'dat', 'die', 'dit', 'deze', 'zij', 'hij',
            'het', 'ik', 'jij', 'je', 'wij', 'we', 'jullie', 'zij', 'ze', 'mij', 'me', 'jou',
            'hem', 'haar', 'ons', 'hun', 'hen', 'is', 'zijn', 'was', 'waren', 'ben', 'bent',
            'heeft', 'hebben', 'had', 'hadden', 'kan', 'kunnen', 'kon', 'konden', 'moet',
            'moeten', 'moest', 'moesten', 'wil', 'willen', 'wilde', 'wilden', 'zal', 'zullen',
            'zou', 'zouden', 'niet', 'geen', 'wel', 'ook', 'nog', 'al', 'reeds', 'maar',
            'of', 'want', 'omdat', 'als', 'toen', 'dan', 'waar', 'wat', 'wie', 'hoe', 'waarom',
            
            # Numbers and quantities
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            'first', 'second', 'third', 'fourth', 'fifth', 'last', 'next', 'previous',
            'million', 'billion', 'trillion', 'thousand', 'hundred', 'dozen',
            'eins', 'zwei', 'drei', 'vier', 'fünf', 'sechs', 'sieben', 'acht', 'neun', 'zehn',
            'erste', 'zweite', 'dritte', 'vierte', 'fünfte', 'letzte', 'nächste',
            'millionen', 'milliarden', 'tausend', 'hundert',
            
            # Time-related words (all languages)
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
            'september', 'october', 'november', 'december', 'morning', 'afternoon', 'evening',
            'night', 'week', 'month', 'year', 'years', 'day', 'days', 'hour', 'hours',
            'minute', 'minutes', 'second', 'seconds', 'time', 'times', 'today', 'yesterday',
            'tomorrow', 'now', 'then', 'later', 'earlier', 'soon', 'recently', 'currently',
            'montag', 'dienstag', 'mittwoch', 'donnerstag', 'freitag', 'samstag', 'sonntag',
            'januar', 'februar', 'märz', 'april', 'mai', 'juni', 'juli', 'august',
            'september', 'oktober', 'november', 'dezember', 'morgen', 'nachmittag', 'abend',
            'nacht', 'woche', 'monat', 'jahr', 'jahre', 'tag', 'tage', 'stunde', 'stunden',
            'lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche',
            'janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août',
            'septembre', 'octobre', 'novembre', 'décembre', 'matin', 'midi', 'soir',
            
            # Common verbs that don't carry meaning
            'about', 'above', 'across', 'after', 'again', 'against', 'along', 'among', 'around',
            'because', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond',
            'during', 'except', 'inside', 'instead', 'outside', 'through', 'throughout',
            'together', 'toward', 'towards', 'under', 'until', 'upon', 'within', 'without',
            'other', 'another', 'each', 'every', 'many', 'much', 'more', 'most', 'some',
            'any', 'few', 'little', 'both', 'either', 'neither', 'several', 'enough',
            'quite', 'rather', 'really', 'very', 'too', 'so', 'such', 'just', 'only',
            'even', 'still', 'yet', 'already', 'almost', 'perhaps', 'maybe', 'probably',
            'certainly', 'definitely', 'absolutely', 'completely', 'totally', 'entirely',
            'exactly', 'especially', 'particularly', 'generally', 'usually', 'normally',
            'actually', 'basically', 'simply', 'clearly', 'obviously', 'however', 'therefore'
        ])
        
        words = text_data.split()
        clean_words = [
            word for word in words 
            if word not in enhanced_stopwords 
            and len(word) > 3 
            and word.isalpha() 
        ]
        
        # Count word frequencies
        word_counts = Counter(clean_words)
        top_words = word_counts.most_common(30)  
        
        if top_words:
            words_list, freq_list = zip(*top_words)
            
            fig = plt.figure(figsize=(10, 11))
            ax = fig.add_subplot(111)
            bars = ax.barh(range(len(words_list)), freq_list, color='steelblue')
            
            ax.set_yticks(range(len(words_list)))
            ax.set_yticklabels(words_list, fontsize=15)
            ax.set_xlabel('Frequency', fontsize=18)
            ax.set_title('Top 30 Most Frequent Words\n', fontweight='bold', fontsize=18)
            ax.invert_yaxis()
            
            ax.tick_params(axis='x', labelsize=15)       

            for i, (bar, freq) in enumerate(zip(bars, freq_list)):
                ax.text(freq + 0.1, i, str(freq), 
                       va='center', ha='left', fontweight='bold', fontsize=15)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("No significant words found after filtering")
    
    with col2:
        st.markdown("### Sentiment Analysis")
        
        sentiments = []
        sentiment_scores = []
        
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        
        for headline in translated_headlines:
            if headline:
                try:
                    scores = analyzer.polarity_scores(headline)
                    compound_score = scores['compound']
                    
                    if compound_score >= 0.05:  
                        sentiment = 'Positive'
                    elif compound_score <= -0.05: 
                        sentiment = 'Negative'
                    else:
                        sentiment = 'Neutral'
                    
                    sentiments.append(sentiment)
                    sentiment_scores.append(compound_score)
                except:
                    sentiments.append('Neutral')
                    sentiment_scores.append(0.0)
        
        if sentiments:
            sentiment_counts = Counter(sentiments)
            
            colors = ['#e74c3c', '#f39c12', '#27ae60']  # Red, Orange, Green
            
            ordered_sentiments = ['Negative', 'Neutral', 'Positive']
            ordered_counts = [sentiment_counts.get(s, 0) for s in ordered_sentiments]
            ordered_colors = [colors[i] for i, count in enumerate(ordered_counts) if count > 0]
            ordered_labels = [s for s, count in zip(ordered_sentiments, ordered_counts) if count > 0]
            ordered_values = [count for count in ordered_counts if count > 0]
            
            if ordered_values:  
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)
                wedges, texts, autotexts = ax.pie(
                    ordered_values,
                    labels=ordered_labels,
                    autopct='%1.1f%%',
                    colors=ordered_colors,
                    startangle=90,
                    explode=[0.05] * len(ordered_values)  
                )
                
                ax.set_title('Sentiment Distribution\n', 
                            fontsize=14, fontweight='bold')
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(12)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                if sentiment_scores:
                    avg_sentiment = np.mean(sentiment_scores)
                    st.write(f"Average Sentiment Score: {avg_sentiment:.3f}")
                    
                    if avg_sentiment > 0.1:
                        overall_mood = "Generally Positive"
                    elif avg_sentiment < -0.1:
                        overall_mood = "Generally Negative"
                    else:
                        overall_mood = "Generally Neutral"
                    
        else:
            st.warning("No headlines available for sentiment analysis")
    
    st.markdown("### Word Cloud")
    
    if text_data:
        wordcloud = WordCloud(
            width=1400,
            height=700,
            background_color='white',
            stopwords=enhanced_stopwords,
            max_words=50,
            max_font_size=70,
            min_font_size=10,
            scale=3,
            random_state=42,
            colormap='plasma',  
            relative_scaling='auto',
            prefer_horizontal=0.6
        ).generate(text_data)
        
        fig = plt.figure(figsize=(16, 9), dpi=300)
        ax = fig.add_subplot(111)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'{search_query} News Word Cloud\n', 
                    fontsize=18, fontweight='bold', pad=30)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("No text data available for word cloud generation")
    
    with st.expander(f"Recent Headlines ({min(len(articles), 5)} articles)"):
        for i, article in enumerate(articles[:5]):
            if article.get('title'):
                original = article['title']
                published = article.get('publishedAt', 'Unknown date')
                detected_lang = languages_detected[i] if i < len(languages_detected) else ''
                    
                st.write(f"**Article {i+1} - {detected_lang.upper()}:**")
                st.write(f"{original}")
                st.caption(f"Published: {published}")
                
                if i < min(len(articles), 5) - 1: 
                    st.markdown("---")
        
        if len(articles) > 5:
            st.caption(f"... and {len(articles) - 5} more articles")



def news_analysis():
    search_query = st.text_input(
        "You may search for any company or topic:",
        value="Rheinmetall",
        placeholder="e.g., Tesla, Apple, Bitcoin, AI, etc.",
        help="Enter any search term to analyze news about that topic. Default is 'Rheinmetall'."
    )
    
    if search_query and search_query.strip():
        with st.spinner(f"Fetching and analyzing news for '{search_query}'..."):
            news_data = run_nlp(search_query.strip())
            if news_data:
                display_nlp(news_data, search_query.strip())
            else:
                st.error(f"Could not fetch news data for '{search_query}'.")
    else:
        st.info("Enter a search term above to start analyzing news.")
    
    st.markdown("---")
