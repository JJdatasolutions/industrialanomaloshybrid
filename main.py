"""
Anomalos Institutional 2.0
Architectuur: Hexagonal (Separation of Concerns)
Bevat: Type hinting, geavanceerde foutafhandeling en gescheiden logica-lagen.
"""

import math
import re
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
import requests
import io
import yfinance as yf
import streamlit as st
import plotly.express as px
from google import genai
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from statsmodels.tsa.stattools import adfuller

# --- 1. CONFIGURATIE & CONSTANTEN ---
MARKETS: Dict[str, Dict[str, str]] = {
    "🇺🇸 USA - S&P 500": {"code": "SP500", "benchmark": "SPY", "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"},
    "🇺🇸 USA - S&P 400 (MidCap)": {"code": "SP400", "benchmark": "MDY", "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"},
    "🇺🇸 USA - Nasdaq 100": {"code": "NDX", "benchmark": "QQQ", "wiki": "https://en.wikipedia.org/wiki/Nasdaq-100"}
}

US_SECTOR_MAP: Dict[str, str] = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV',
    'Energy': 'XLE', 'Consumer Discretionary': 'XLY', 'Industrials': 'XLI',
    'Utilities': 'XLU', 'Materials': 'XLB', 'Real Estate': 'XLRE',
    'Communication Services': 'XLC', 'Consumer Staples': 'XLP'
}

COLOR_MAP: Dict[str, str] = {
    "1. LEADING": "#006400", "2. WEAKENING": "#FFA500", 
    "3. LAGGING": "#DC143C", "4. IMPROVING": "#90EE90"
}


# --- 2. DATA ADAPTERS (De stekkertjes naar de buitenwereld) ---
class MarketDataProvider:
    """Verantwoordelijk voor het ophalen van externe ruwe data."""
    
    @staticmethod
    @st.cache_data(ttl=24*3600)
    def get_constituents(market_key: str) -> pd.DataFrame:
        """Haalt de lijst van aandelen op via Wikipedia."""
        try:
            url = MARKETS[market_key]['wiki']
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            tables = pd.read_html(io.StringIO(response.text))
            
            # Zoek de juiste tabel op basis van sleutelwoorden
            target_df = pd.DataFrame()
            for df in tables:
                cols = [str(c).lower() for c in df.columns]
                if any("symbol" in c or "ticker" in c for c in cols):
                    target_df = df
                    break
                    
            if target_df.empty:
                return pd.DataFrame()
                
            # Identificeer de juiste kolommen dynamisch
            cols = target_df.columns
            ticker_col = next((c for c in cols if "symbol" in str(c).lower() or "ticker" in str(c).lower()), None)
            sector_col = next((c for c in cols if "sector" in str(c).lower()), None)
            
            if not ticker_col:
                return pd.DataFrame()

            df_clean = pd.DataFrame()
            df_clean['Ticker'] = target_df[ticker_col].astype(str).str.replace('.', '-', regex=False)
            df_clean['Sector'] = target_df[sector_col] if sector_col else "Unknown"
            
            return df_clean

        except requests.RequestException as e:
            st.error(f"Netwerkfout bij ophalen Wikipedia data: {e}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Onverwachte fout bij dataverwerking: {e}")
            return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_price_data(tickers: List[str], period: str = "1y") -> pd.DataFrame:
        """Haalt historische prijsdata op via yfinance."""
        if not tickers:
            return pd.DataFrame()
        try:
            data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.get_level_values(0):
                    data = data['Close']
                elif 'Close' in data.columns.get_level_values(1):
                    data = data.xs('Close', level=1, axis=1)
            return data
        except Exception as e:
            st.warning(f"Fout bij ophalen prijsdata: {e}")
            return pd.DataFrame()


# --- 3. DOMEIN LOGICA (De wiskundige kern) ---
class FinancialMath:
    """Bevat pure rekenkundige operaties, losgekoppeld van data-extractie."""
    
    @staticmethod
    def calculate_rrg(df_prices: pd.DataFrame, benchmark: str) -> pd.DataFrame:
        """Berekent de Relative Rotation Graph (RRG) statistieken."""
        if df_prices.empty or benchmark not in df_prices.columns:
            return pd.DataFrame()
            
        rrg_records = []
        bench_series = df_prices[benchmark]
        
        for ticker in df_prices.columns:
            if ticker == benchmark:
                continue
                
            try:
                # Vectorgestuurde berekeningen in plaats van loops
                rs = df_prices[ticker] / bench_series
                rs_ma = rs.rolling(100).mean()
                rs_ratio = 100 * (rs / rs_ma)
                rs_mom = 100 * (rs_ratio / rs_ratio.shift(10))
                
                # Verwijder lege waarden (NaN) veilig
                valid_data = rs_ratio.dropna()
                if valid_data.empty:
                    continue
                    
                curr_r = rs_ratio.iloc[-1]
                curr_m = rs_mom.iloc[-1]
                
                # Wiskundige plaatsing
                dx, dy = curr_r - 100, curr_m - 100
                dist = math.hypot(dx, dy)
                heading_deg = (math.degrees(math.atan2(dy, dx)) + 360) % 360
                
                # Kwadrant logica
                if curr_r > 100 and curr_m > 100: status = "1. LEADING"
                elif curr_r < 100 and curr_m > 100: status = "4. IMPROVING"
                elif curr_r < 100 and curr_m < 100: status = "3. LAGGING"
                else: status = "2. WEAKENING"
                
                is_candidate = (status in ["1. LEADING", "4. IMPROVING"]) and (0 <= heading_deg <= 90)
                
                rrg_records.append({
                    'Ticker': ticker, 'Kwadrant': status, 'RS-Ratio': curr_r,
                    'RS-Momentum': curr_m, 'Distance': dist, 'Heading': heading_deg,
                    'Candidate': is_candidate
                })
            except Exception:
                # We negeren individuele ticker fouten (bijv. te weinig data) om de rest niet te blokkeren
                continue
                
        return pd.DataFrame(rrg_records)


class QuantitativeEngine:
    """Machine Learning modellen voor risico- en rendementsanalyse."""
    
    def __init__(self) -> None:
        self.lookback = 30
        self.rf = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
        self.meta_labeler = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)

    @staticmethod
    def _make_stationary(series: np.ndarray) -> np.ndarray:
        """Zorgt dat een tijdreeks stationair is voor ML-invoer."""
        series_clean = np.nan_to_num(series)
        if len(series_clean) < 30:
            return series_clean
        if adfuller(series_clean)[1] > 0.05:
            return np.insert(np.diff(series_clean), 0, 0)
        return series_clean

    def analyze(self, price_series: pd.Series, vix_data: np.ndarray, term_spread: np.ndarray) -> Tuple[float, float]:
        """Analyseert een aandeel en geeft signaal en betrouwbaarheid (bet_size) terug."""
        try:
            prices = price_series.dropna().values
            if len(prices) < self.lookback + 20:
                return 0.0, 0.0
                
            p_stat = self._make_stationary(prices)
            v_stat = self._make_stationary(vix_data[-len(prices):])
            t_stat = self._make_stationary(term_spread[-len(prices):])
            
            X, y = [], []
            for i in range(len(p_stat) - self.lookback - 1):
                X.append([
                    np.mean(p_stat[i:i+self.lookback]), np.std(p_stat[i:i+self.lookback]),
                    np.mean(v_stat[i:i+self.lookback]), np.mean(t_stat[i:i+self.lookback])
                ])
                y.append(p_stat[i+self.lookback+1] - p_stat[i+self.lookback])
                
            if len(X) < 30:
                return 0.0, 0.0
                
            split = int(len(X) * 0.8)
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split:], y[split:]
            
            self.rf.fit(X_train, y_train)
            preds = self.rf.predict(X_test)
            
            y_meta = [1 if (p > 0 and r > 0) or (p < 0 and r < 0) else 0 for p, r in zip(preds, y_test)]
            self.meta_labeler.fit(X_test, y_meta)
            
            curr_feat = [[
                np.mean(p_stat[-self.lookback:]), np.std(p_stat[-self.lookback:]),
                np.mean(v_stat[-self.lookback:]), np.mean(t_stat[-self.lookback:])
            ]]
            
            signal = self.rf.predict(curr_feat)[0]
            bet_size = self.meta_labeler.predict_proba(curr_feat)[0][1]
            
            return float(signal), float(bet_size)
            
        except Exception as e:
            # Fouten worden opgevangen en als neuraal resultaat (0) doorgegeven
            return 0.0, 0.0


class AIAgentAdapter:
    """Adapter voor communicatie met Google Gemini (zonder asyncio hacks)."""
    
    @staticmethod
    def run_swarm_debate(ticker: str, context: str, api_key: str, model_name: str = "gemini-2.0-flash") -> Dict[str, Any]:
        """Gebruikt de Gemini API op een veilige, synchrone manier."""
        try:
            client = genai.Client(api_key=api_key)
            prompt = (
                f"Je bent een Expert Financieel Comité (Momentum, Fundamenteel, Risk). "
                f"Analyseer ticker {ticker}. Context: {context}. "
                f"Geef EXACT en ALLEEN dit JSON formaat terug: {{\"rationale\": \"uitleg\", \"score\": 5}} "
                f"Score is tussen -10 en +10. De Risk Manager heeft het laatste woord."
            )
            
            # Synchrone oproep: veilig voor Streamlit Cloud
            response = client.models.generate_content(model=model_name, contents=prompt)
            txt = response.text
            
            # Extract JSON veilig
            json_match = re.search(r'\{.*\}', txt, re.DOTALL)
            if json_match:
                import json
                return json.loads(json_match.group())
            return {"rationale": "Ongeldig AI antwoord (geen JSON).", "score": 0}
            
        except Exception as e:
            return {"rationale": f"API Error: {str(e)}", "score": 0}


# --- 4. PRESENTATIE LAAG (Streamlit UI) ---
def render_ui() -> None:
    """De hoofdgebruikersinterface (De 'Ober')."""
    st.set_page_config(page_title="Anomalos Institutional 2.0", layout="wide", page_icon="🦅")
    
    if 'rrg_candidates' not in st.session_state:
        st.session_state['rrg_candidates'] = []

    st.sidebar.title("🦅 Anomalos Pro")
    api_key = st.sidebar.text_input("Google Gemini API Key", type="password")
    market_sel = st.sidebar.selectbox("Markt", list(MARKETS.keys()))
    sector_sel = st.sidebar.selectbox("Filter Sector", ["Alle"] + list(US_SECTOR_MAP.keys()))

    tab_screener, tab_execution = st.tabs(["1. RRG Screener (Retriever)", "2. Anomalos Execution (Agent)"])

    # --- TAB 1 ---
    with tab_screener:
        st.markdown("### 🔍 Stap 1: De Alpha-Filter Laag")
        if st.button("Start Screening"):
            with st.spinner("Markt scannen... Dit kan even duren."):
                constituents = MarketDataProvider.get_constituents(market_sel)
                
                if constituents.empty:
                    st.error("Fout bij ophalen marktonderdelen.")
                    return
                    
                if sector_sel != "Alle" and 'Sector' in constituents.columns:
                    tickers = constituents[constituents['Sector'] == sector_sel]['Ticker'].tolist()
                else:
                    tickers = constituents['Ticker'].head(80).tolist()
                    
                benchmark = MARKETS[market_sel]['benchmark']
                tickers = list(set(tickers + [benchmark]))
                
                df_prices = MarketDataProvider.get_price_data(tickers)
                
                if df_prices.empty:
                    st.error("Kon geen prijsdata ophalen via Yahoo Finance.")
                    return
                    
                rrg_df = FinancialMath.calculate_rrg(df_prices, benchmark)
                
                if rrg_df.empty:
                    st.error("RRG Berekening mislukt: te weinig historische data.")
                else:
                    # Plotly Grafiek
                    fig = px.scatter(
                        rrg_df, x="RS-Ratio", y="RS-Momentum", color="Kwadrant", 
                        text="Ticker", hover_data=["Heading", "Candidate"],
                        color_discrete_map=COLOR_MAP, height=600
                    )
                    fig.add_hline(y=100, line_dash="dash", line_color="grey")
                    fig.add_vline(x=100, line_dash="dash", line_color="grey")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    candidates = rrg_df[rrg_df['Candidate']].sort_values('Distance', ascending=False)
                    if not candidates.empty:
                        st.success(f"{len(candidates)} kandidaten gevonden.")
                        st.dataframe(candidates[['Ticker', 'Kwadrant', 'Heading', 'Distance']])
                        st.session_state['rrg_candidates'] = candidates['Ticker'].head(10).tolist()
                        st.session_state['price_data'] = df_prices
                    else:
                        st.warning("Geen geschikte kandidaten gevonden in de geselecteerde markt.")

    # --- TAB 2 ---
    with tab_execution:
        st.markdown("### 🧠 Stap 2: Multi-Agent Validatie")
        candidates = st.session_state.get('rrg_candidates', [])
        
        if not candidates:
            st.warning("Draai eerst de Screener in Tab 1.")
            return
            
        st.write(f"**Geselecteerde Shortlist:** {', '.join(candidates)}")
        
        if st.button("🚀 Activeer Anomalos Agents"):
            if not api_key:
                st.error("API Key is vereist in het zijmenu.")
                return
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            quant_eng = QuantitativeEngine()
            df_prices = st.session_state['price_data']
            
            # Voorbeeld mock data voor macro features (in productie vervangen door echte data)
            vix_mock = np.full(100, 20.0)
            term_mock = np.full(100, 1.5)
            
            for idx, t in enumerate(candidates):
                status_text.markdown(f"🕵️ **Analyseren:** {t}...")
                progress_bar.progress((idx + 1) / len(candidates))
                
                # 1. Quant Check
                if t in df_prices.columns:
                    sig, bet = quant_eng.analyze(df_prices[t], vix_mock, term_mock)
                else:
                    sig, bet = 0.0, 0.0
                    
                # 2. AI Check (Synchroon)
                if bet > 0.50:  # Iets verlaagd voor demo-doeleinden
                    context = f"Laatste prijs: {df_prices[t].iloc[-1]:.2f}" if t in df_prices.columns else "Geen data"
                    ai_res = AIAgentAdapter.run_swarm_debate(t, context, api_key)
                    
                    # 3. Score Berekening
                    ai_score_norm = (ai_res.get('score', 0) + 10) / 20 
                    final_conviction = (bet * 0.7) + (ai_score_norm * 0.3)
                    
                    if final_conviction > 0.5:
                        results.append({
                            'Ticker': t, 'Conviction': round(final_conviction, 2),
                            'Quant_Conf': round(bet, 2), 'AI_Score': ai_res.get('score', 0),
                            'Rationale': ai_res.get('rationale', '')
                        })
            
            status_text.markdown("✅ Analyse voltooid.")
            
            if results:
                res_df = pd.DataFrame(results).sort_values('Conviction', ascending=False)
                st.dataframe(res_df[['Ticker', 'Conviction', 'AI_Score', 'Quant_Conf']], use_container_width=True)
                
                for _, row in res_df.iterrows():
                    with st.expander(f"{row['Ticker']} - Rationale"):
                        st.write(row['Rationale'])
            else:
                st.warning("Geen enkel aandeel kwam door de strenge selectie.")

# Zorgt ervoor dat het script alleen draait als het direct wordt gestart
if __name__ == "__main__":
    render_ui()
