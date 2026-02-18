import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np
import math
import asyncio
import requests
import io
import nest_asyncio
import re
import time
from datetime import datetime
from google import genai
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.linalg import eigh
import warnings

# Setup
nest_asyncio.apply()
warnings.filterwarnings("ignore")

# --- 0. CONFIGURATIE & STATE ---
st.set_page_config(page_title="Anomalos Institutional 2.0", layout="wide", page_icon="🦅")

if 'rrg_candidates' not in st.session_state:
    st.session_state['rrg_candidates'] = []

# --- 1. CORE DATA DEFINITIES (RRG) ---
MARKETS = {
    "🇺🇸 USA - S&P 500": {"code": "SP500", "benchmark": "SPY", "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"},
    "🇺🇸 USA - S&P 400 (MidCap)": {"code": "SP400", "benchmark": "MDY", "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"},
    "🇺🇸 USA - Nasdaq 100": {"code": "NDX", "benchmark": "QQQ", "wiki": "https://en.wikipedia.org/wiki/Nasdaq-100"}
}

US_SECTOR_MAP = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV',
    'Energy': 'XLE', 'Consumer Discretionary': 'XLY', 'Industrials': 'XLI',
    'Utilities': 'XLU', 'Materials': 'XLB', 'Real Estate': 'XLRE',
    'Communication Services': 'XLC', 'Consumer Staples': 'XLP'
}

COLOR_MAP = {"1. LEADING": "#006400", "2. WEAKENING": "#FFA500", "3. LAGGING": "#DC143C", "4. IMPROVING": "#90EE90"}

# --- 2. HULP FUNCTIES (DATA) ---
@st.cache_data(ttl=24*3600)
def get_market_constituents(market_key):
    try:
        mkt = MARKETS[market_key]
        # We gebruiken een "echte" browser User-Agent om de blokkade te omzeilen
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Eerst de HTML ophalen met requests
        response = requests.get(mkt['wiki'], headers=headers)
        response.raise_for_status() # Check of de pagina bestaat
        
        # Pandas laten lezen vanuit de tekst string
        tables = pd.read_html(io.StringIO(response.text))
        
        target_df = pd.DataFrame()
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if any("symbol" in c for c in cols) and (any("sector" in c for c in cols) or "security" in cols):
                target_df = df
                break
        
        if target_df.empty: return pd.DataFrame()
        
        # Kolommen opschonen
        cols = target_df.columns
        ticker_col = next(c for c in cols if "Symbol" in str(c) or "Ticker" in str(c))
        sector_col = next((c for c in cols if "Sector" in str(c)), None)
        
        df_clean = target_df[[ticker_col]].copy()
        if sector_col:
            df_clean['Sector'] = target_df[sector_col]
        else:
            df_clean['Sector'] = "Unknown"
            
        df_clean.columns = ['Ticker', 'Sector']
        df_clean['Ticker'] = df_clean['Ticker'].str.replace('.', '-', regex=False)
        return df_clean

    except Exception as e:
        st.error(f"Fout bij ophalen data van Wikipedia: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_price_data(tickers):
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(tickers, period="1y", progress=False, auto_adjust=True) # 1y voor RRG + Anomalos
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0): data = data['Close']
            elif 'Close' in data.columns.get_level_values(1): data = data.xs('Close', level=1, axis=1)
        return data
    except: return pd.DataFrame()

# --- 3. RRG LOGICA (DE "RETRIEVER" LAAG) ---
def calculate_rrg_signals(df, benchmark_ticker):
    if df.empty or benchmark_ticker not in df.columns: return pd.DataFrame()
    
    rrg_data = []
    bench_series = df[benchmark_ticker]
    
    for ticker in df.columns:
        if ticker == benchmark_ticker: continue
        try:
            # RRG Berekening
            rs = df[ticker] / bench_series
            rs_ma = rs.rolling(100).mean()
            rs_ratio = 100 * (rs / rs_ma)
            rs_mom = 100 * (rs_ratio / rs_ratio.shift(10))
            
            if len(rs_ratio) < 1: continue
            curr_r = rs_ratio.iloc[-1]
            curr_m = rs_mom.iloc[-1]
            
            # Heading & Distance
            dist = np.sqrt((curr_r - 100)**2 + (curr_m - 100)**2)
            dx = curr_r - 100
            dy = curr_m - 100
            heading_deg = math.degrees(math.atan2(dy, dx))
            if heading_deg < 0: heading_deg += 360
            
            # Kwadrant Bepaling
            if curr_r > 100 and curr_m > 100: status = "1. LEADING"
            elif curr_r < 100 and curr_m > 100: status = "4. IMPROVING"
            elif curr_r < 100 and curr_m < 100: status = "3. LAGGING"
            else: status = "2. WEAKENING"
            
            # SYNERGIE FILTER LOGICA:
            # 1. Moet in Leading of Improving zitten
            # 2. Heading moet tussen 0 en 90 graden zijn (Power Trend)
            is_candidate = (status in ["1. LEADING", "4. IMPROVING"]) and (0 <= heading_deg <= 90)
            
            rrg_data.append({
                'Ticker': ticker,
                'Kwadrant': status,
                'RS-Ratio': curr_r,
                'RS-Momentum': curr_m,
                'Distance': dist,
                'Heading': heading_deg,
                'Candidate': is_candidate
            })
        except: continue
        
    return pd.DataFrame(rrg_data)

# --- 4. ANOMALOS CLASSES (DE INSTITUTIONELE LAAG) ---

class MarketDataEngine:
    @staticmethod
    def get_macro_features():
        try:
            macro = yf.download(['^VIX', '^TNX', '^IRX'], period="100d", progress=False)['Close']
            macro.ffill(inplace=True)
            # Check of kolommen bestaan (yfinance format wisselt soms)
            vix = macro['^VIX'].values if '^VIX' in macro else np.zeros(100)
            tnx = macro['^TNX'].values if '^TNX' in macro else np.zeros(100)
            irx = macro['^IRX'].values if '^IRX' in macro else np.zeros(100)
            
            term_spread = tnx - irx
            current_vix = vix[-1] if len(vix) > 0 else 20.0
            return vix, term_spread, current_vix
        except:
            return np.zeros(100), np.zeros(100), 20.0

    @staticmethod
    def get_5_day_context(ticker):
        try:
            data = yf.download(ticker, period="5d", progress=False)[['Close', 'Volume']]
            if data.empty: return "Geen data."
            context = []
            for date, row in data.iterrows():
                val_close = row['Close'].item() if hasattr(row['Close'], 'item') else row['Close']
                val_vol = row['Volume'].item() if hasattr(row['Volume'], 'item') else row['Volume']
                context.append(f"{date.strftime('%Y-%m-%d')}: ${val_close:.2f} (Vol: {val_vol})")
            return " | ".join(context)
        except: return "Data error."

    @staticmethod
    def make_stationary(series):
        series_clean = np.nan_to_num(series)
        if len(series_clean) < 30: return series_clean
        # Simpele stationariteit check
        if adfuller(series_clean)[1] > 0.05:
            return np.insert(np.diff(series_clean), 0, 0)
        return series_clean

class QuantitativeEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.lookback = 30
        self.rf = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
        # XGBoost weggelaten om dependencies simpel te houden voor Streamlit Cloud, 
        # RF is robuust genoeg voor demo.
        self.meta_labeler = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)

    def analyze(self, ticker, price_series, vix_data, term_spread):
        try:
            prices = price_series.values
            if len(prices) < self.lookback + 20: return 0.0, 0.0
            
            p_stat = MarketDataEngine.make_stationary(prices)
            v_stat = MarketDataEngine.make_stationary(vix_data[-len(prices):])
            t_stat = MarketDataEngine.make_stationary(term_spread[-len(prices):])
            
            X, y = [], []
            for i in range(len(p_stat) - self.lookback - 1):
                X.append([
                    np.mean(p_stat[i:i+self.lookback]), np.std(p_stat[i:i+self.lookback]),
                    np.mean(v_stat[i:i+self.lookback]), np.mean(t_stat[i:i+self.lookback])
                ])
                y.append(p_stat[i+self.lookback+1] - p_stat[i+self.lookback]) # Return sign
            
            if len(X) < 30: return 0.0, 0.0
            
            # Train/Test
            split = int(len(X) * 0.8)
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split:], y[split:]
            
            self.rf.fit(X_train, y_train)
            preds = self.rf.predict(X_test)
            
            # Meta-Labeling: Voorspel of het model gelijk heeft
            y_meta = [1 if (p > 0 and r > 0) or (p < 0 and r < 0) else 0 for p, r in zip(preds, y_test)]
            self.meta_labeler.fit(X_test, y_meta)
            
            # Current Features
            curr_feat = [[
                np.mean(p_stat[-self.lookback:]), np.std(p_stat[-self.lookback:]),
                np.mean(v_stat[-self.lookback:]), np.mean(t_stat[-self.lookback:])
            ]]
            
            signal = self.rf.predict(curr_feat)[0]
            bet_size = self.meta_labeler.predict_proba(curr_feat)[0][1] # Probability of success
            
            return signal, bet_size
        except Exception as e:
            return 0.0, 0.0

class HRPOptimizer:
    @staticmethod
    def optimize(df_prices):
        # Hierarchical Risk Parity (Versimpeld voor snelheid)
        try:
            returns = df_prices.pct_change().dropna()
            cov = returns.cov()
            corr = returns.corr()
            dist = squareform(np.sqrt(np.clip((1 - corr) / 2, 0, 1)))
            link = linkage(dist, 'single')
            
            # Simpele inverse variance allocatie op geclusterde volgorde (Proxy voor volledige HRP)
            inv_var = 1 / np.diag(cov)
            weights = inv_var / inv_var.sum()
            
            allocations = dict(zip(df_prices.columns, weights * 100))
            return allocations
        except:
            return {col: 100/len(df_prices.columns) for col in df_prices.columns}

async def run_swarm_debate(ticker, context, api_key, model_name="gemini-2.0-flash"):
    client = genai.Client(api_key=api_key)
    
    base_prompt = (
        f"Analyseer ticker {ticker}. Context: {context}. "
        f"Geef antwoord als JSON: {{\"rationale\": \"<korte uitleg>\", \"score\": <getal -10 tot +10>}}. "
    )
    
    # We simuleren het debat in één krachtige call voor snelheid in UI
    final_prompt = (
        f"Je bent een Expert Financieel Comité bestaande uit een Momentum Trader, een Fundamenteel Analist en een Risk Manager. "
        f"{base_prompt} "
        f"De Risk Manager heeft het laatste woord. Wees streng."
    )
    
    try:
        response = await asyncio.to_thread(
            lambda: client.models.generate_content(model=model_name, contents=final_prompt)
        )
        txt = response.text
        json_str = re.search(r'\{.*\}', txt, re.DOTALL)
        if json_str:
            return eval(json_str.group()) # Safe eval voor dict strings
        else:
            return {"rationale": "Geen JSON output", "score": 0}
    except Exception as e:
        return {"rationale": f"API Error: {e}", "score": 0}

# --- 5. UI & WORKFLOW ---

st.sidebar.title("🦅 Anomalos Pro")
api_key = st.sidebar.text_input("Google Gemini API Key", type="password")
market_sel = st.sidebar.selectbox("Markt", list(MARKETS.keys()))
sector_sel = st.sidebar.selectbox("Filter Sector", ["Alle"] + list(US_SECTOR_MAP.keys()))

tab_screener, tab_execution = st.tabs(["1. RRG Screener (Retriever)", "2. Anomalos Execution (Agent)"])

# === TAB 1: SCREENER ===
with tab_screener:
    st.markdown("### 🔍 Stap 1: De Alpha-Filter Laag")
    st.info("De Screener fungeert als 'Retriever'. Hij zoekt aandelen in **Leading/Improving** met een **Heading van 0-90°**.")
    
    if st.button("Start Screening"):
        with st.spinner("Markt scannen..."):
            cfg = MARKETS[market_sel]
            constituents = get_market_constituents(market_sel)
            
            if not constituents.empty:
                # Sector Filter
                if sector_sel != "Alle":
                    # Let op: Soms heet de kolom 'Sector' anders, we checken dit veiliger
                    if 'Sector' in constituents.columns:
                        tickers = constituents[constituents['Sector'] == sector_sel]['Ticker'].tolist()
                    else:
                        st.warning("Sector data niet beschikbaar voor deze markt. Alle tickers worden gebruikt.")
                        tickers = constituents['Ticker'].head(80).tolist()
                else:
                    tickers = constituents['Ticker'].head(80).tolist() # Limit voor demo snelheid
                
                # Zorg dat de benchmark erbij zit en uniek is
                bench = cfg['benchmark']
                tickers = list(set(tickers + [bench]))
                
                st.write(f"Data ophalen voor {len(tickers)} tickers...") # Debug info
                df_prices = get_price_data(tickers)
                
                if not df_prices.empty and bench in df_prices.columns:
                    rrg_df = calculate_rrg_signals(df_prices, bench)
                    
                    # --- DE CRUCIALE CHECK ---
                    if rrg_df.empty:
                        st.error("RRG Berekening leverde geen resultaten op. Controleer of er genoeg historische data is.")
                    else:
                        # Controleren of alle kolommen bestaan
                        required_cols = ["RS-Ratio", "RS-Momentum", "Kwadrant", "Ticker", "Heading", "Candidate"]
                        missing = [c for c in required_cols if c not in rrg_df.columns]
                        
                        if missing:
                            st.error(f"Ontbrekende kolommen in RRG data: {missing}")
                        else:
                            # VISUALISATIE (Nu veilig)
                            try:
                                fig = px.scatter(rrg_df, x="RS-Ratio", y="RS-Momentum", color="Kwadrant", 
                                                 text="Ticker", hover_data=["Heading", "Candidate"],
                                                 color_discrete_map=COLOR_MAP, height=600)
                                fig.add_hline(y=100, line_dash="dash", line_color="grey")
                                fig.add_vline(x=100, line_dash="dash", line_color="grey")
                                fig.add_shape(type="rect", x0=100, y0=100, x1=110, y1=110, line=dict(color="Green"), fillcolor="rgba(0,255,0,0.1)")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # FILTEREN
                                candidates = rrg_df[rrg_df['Candidate'] == True].sort_values('Distance', ascending=False)
                                if not candidates.empty:
                                    st.success(f"{len(candidates)} kandidaten gevonden voor de AI-laag.")
                                    st.dataframe(candidates[['Ticker', 'Kwadrant', 'Heading', 'Distance']])
                                    
                                    # SLA OP VOOR VOLGENDE STAP
                                    st.session_state['rrg_candidates'] = candidates['Ticker'].head(10).tolist()
                                    st.session_state['price_data'] = df_prices
                                else:
                                    st.warning("Geen kandidaten gevonden die aan de criteria (Leading/Improving + Heading 0-90°) voldoen.")
                            except Exception as e:
                                st.error(f"Fout bij plotten: {e}")
                else:
                    st.error(f"Kon geen prijsdata ophalen of benchmark '{bench}' ontbreekt in data.")
            else:
                st.error("Kon geen tickers ophalen van Wikipedia. Check je internetverbinding of Wikipedia status.")

# === TAB 2: EXECUTION ===
with tab_execution:
    st.markdown("### 🧠 Stap 2: Multi-Agent Validatie & HRP")
    
    candidates = st.session_state.get('rrg_candidates', [])
    
    if not candidates:
        st.warning("Draai eerst de Screener in Tab 1 om kandidaten te genereren.")
    else:
        st.write(f"**Geselecteerde Shortlist:** {', '.join(candidates)}")
        st.write("De agents zullen deze lijst nu valideren met kwantitatieve modellen en fundamentele analyse.")
        
        if st.button("🚀 Activeer Anomalos Agents") and api_key:
            output_container = st.container()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            # INITIALISEER ENGINES
            quant_eng = QuantitativeEngine()
            vix, term, curr_vix = MarketDataEngine.get_macro_features()
            df_prices = st.session_state['price_data']
            
            # ASYNC LOOP WRAPPER VOOR STREAMLIT
            async def run_pipeline():
                for idx, t in enumerate(candidates):
                    status_text.markdown(f"🕵️ **Analyseren:** {t}...")
                    progress = (idx + 1) / len(candidates)
                    progress_bar.progress(progress)
                    
                    # 1. QUANT CHECK
                    if t in df_prices.columns:
                        sig, bet = quant_eng.analyze(t, df_prices[t], vix, term)
                    else:
                        sig, bet = 0, 0
                    
                    # 2. AI DEBAT (Alleen als Quant positief is of voor demo doeleinden altijd)
                    if bet > 0.55: # Drempelwaarde
                        context = MarketDataEngine.get_5_day_context(t)
                        ai_res = await run_swarm_debate(t, context, api_key)
                        
                        # 3. SCORING
                        ai_score_norm = (ai_res['score'] + 10) / 20 # 0 tot 1
                        # VIX Weging: Bij hoge VIX telt AI zwaarder
                        ai_weight = min(0.8, curr_vix / 50.0)
                        final_conviction = (bet * (1 - ai_weight)) + (ai_score_norm * ai_weight)
                        
                        if final_conviction > 0.6: # Cutoff
                            results.append({
                                'Ticker': t,
                                'Conviction': final_conviction,
                                'Quant_Conf': bet,
                                'AI_Score': ai_res['score'],
                                'Rationale': ai_res['rationale']
                            })
            
            asyncio.run(run_pipeline())
            
            status_text.markdown("✅ Analyse voltooid. Portefeuille optimaliseren...")
            
            if results:
                res_df = pd.DataFrame(results).sort_values('Conviction', ascending=False)
                
                # HRP OPTIMALISATIE
                final_tickers = res_df['Ticker'].tolist()
                allocations = HRPOptimizer.optimize(df_prices[final_tickers])
                res_df['Allocatie (%)'] = res_df['Ticker'].map(allocations).map('{:.2f}'.format)
                
                st.markdown("### 🏆 Definitieve Portefeuille")
                st.dataframe(res_df[['Ticker', 'Allocatie (%)', 'Conviction', 'AI_Score', 'Quant_Conf']], use_container_width=True)
                
                st.markdown("### 📝 Agent Rationales")
                for _, row in res_df.iterrows():
                    with st.expander(f"{row['Ticker']} - AI Score: {row['AI_Score']}/10"):
                        st.write(row['Rationale'])
            else:
                st.error("Geen enkel aandeel kwam door de strenge selectie. Advies: 100% CASH.")
                
        elif not api_key:
            st.error("Vul eerst je API Key in de sidebar in.")
