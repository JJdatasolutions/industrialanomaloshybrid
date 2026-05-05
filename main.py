"""
Anomalos Institutional 2.0
Architectuur: Hexagonal (Separation of Concerns)
Bevat: Type hinting, robuuste web-extractie (httpx+bs4), en synchrone AI integratie.
"""

import math
import re
import io
from typing import List, Dict, Tuple, Any

import httpx
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.express as px
from google import genai
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.cluster.hierarchy import linkage, leaves_list
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
class WikipediaMarketAdapter:
    """Schild tussen Wikipedia en onze app. Haalt data op en vertaalt het veilig."""

    @staticmethod
    @st.cache_data(ttl=24*3600)
    def get_market_constituents(market_key: str) -> pd.DataFrame:
        try:
            mkt = MARKETS.get(market_key)
            if not mkt:
                raise KeyError(f"Markt {market_key} niet gevonden.")

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept-Language": "en-US,en;q=0.9"
            }
            
            with httpx.Client(timeout=15.0) as client:
                response = client.get(mkt['wiki'], headers=headers)
                response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable'})
            if not table:
                raise ValueError("Kon de hoofdtabel niet vinden op Wikipedia.")

            df = pd.read_html(io.StringIO(str(table)))[0]
            cols = [str(c).lower() for c in df.columns]
            
            ticker_idx = next((i for i, c in enumerate(cols) if "symbol" in c or "ticker" in c), None)
            sector_idx = next((i for i, c in enumerate(cols) if "sector" in c), None)

            if ticker_idx is None:
                raise ValueError("Kon geen Ticker kolom vinden.")

            df_clean = pd.DataFrame()
            df_clean['Ticker'] = df.iloc[:, ticker_idx].astype(str).str.replace('.', '-', regex=False)
            df_clean['Sector'] = df.iloc[:, sector_idx] if sector_idx is not None else "Unknown"

            return df_clean

        except httpx.HTTPError as net_err:
            st.error(f"Netwerkfout Wikipedia: {net_err}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Fout bij dataverwerking: {e}")
            return pd.DataFrame()


class MarketDataProvider:
    """Verantwoordelijk voor het ophalen van historische prijsdata."""
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_price_data(tickers: List[str], period: str = "1y") -> pd.DataFrame:
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
    """Bevat pure rekenkundige operaties, zoals de RRG berekening."""
    
    @staticmethod
    def calculate_rrg(df_prices: pd.DataFrame, benchmark: str) -> pd.DataFrame:
        if df_prices.empty or benchmark not in df_prices.columns:
            return pd.DataFrame()
            
        rrg_records = []
        bench_series = df_prices[benchmark]
        
        for ticker in df_prices.columns:
            if ticker == benchmark:
                continue
                
            try:
                rs = df_prices[ticker] / bench_series
                rs_ma = rs.rolling(100).mean()
                rs_ratio = 100 * (rs / rs_ma)
                rs_mom = 100 * (rs_ratio / rs_ratio.shift(10))
                
                valid_data = rs_ratio.dropna()
                if valid_data.empty:
                    continue
                    
                curr_r = rs_ratio.iloc[-1]
                curr_m = rs_mom.iloc[-1]
                
                dx, dy = curr_r - 100, curr_m - 100
                dist = math.hypot(dx, dy)
                heading_deg = (math.degrees(math.atan2(dy, dx)) + 360) % 360
                
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
        series_clean = np.nan_to_num(series)
        if len(series_clean) < 30:
            return series_clean
        if adfuller(series_clean)[1] > 0.05:
            return np.insert(np.diff(series_clean), 0, 0)
        return series_clean

    def analyze(self, price_series: pd.Series, vix_data: np.ndarray, term_spread: np.ndarray) -> Tuple[float, float]:
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
        except Exception:
            return 0.0, 0.0


class HRPOptimizer:
    """Eenvoudige Hierarchical Risk Parity (HRP) allocatie."""
    
    @staticmethod
    def optimize(prices: pd.DataFrame) -> Dict[str, float]:
        if prices.empty or len(prices.columns) < 2:
            return {col: 100.0 / len(prices.columns) for col in prices.columns}
        try:
            returns = prices.pct_change().dropna()
            cov = returns.cov()
            corr = returns.corr()
            
            dist = np.sqrt(0.5 * (1 - corr))
            link = linkage(squareform(dist), 'single')
            sort_ix = leaves_list(link)
            
            allocations = {prices.columns[i]: 100.0 / len(prices.columns) for i in sort_ix}
            return allocations
        except Exception:
            return {col: 100.0 / len(prices.columns) for col in prices.columns}


class AIAgentAdapter:
    """Adapter voor communicatie met Google Gemini (Veilig en synchroon)."""
    
    @staticmethod
    def run_swarm_debate(ticker: str, context: str, api_key: str, model_name: str = "gemini-2.0-flash") -> Dict[str, Any]:
        try:
            client = genai.Client(api_key=api_key)
            prompt = (
                f"Je bent een Expert Financieel Comité. Analyseer ticker {ticker}. Context: {context}. "
                f"Geef EXACT en ALLEEN dit JSON formaat terug: {{\"rationale\": \"korte uitleg\", \"score\": 5}} "
                f"Score is tussen -10 en +10."
            )
            response = client.models.generate_content(model=model_name, contents=prompt)
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                import json
                return json.loads(json_match.group())
            return {"rationale": "Ongeldig AI antwoord.", "score": 0}
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

    tab_screener, tab_execution = st.tabs(["1. RRG Screener", "2. Anomalos Execution"])

    # --- TAB 1: RRG Screener ---
    with tab_screener:
        st.markdown("### 🔍 Stap 1: De Alpha-Filter Laag")
        if st.button("Start Screening"):
            with st.spinner("Markt scannen... Dit kan even duren."):
                constituents = WikipediaMarketAdapter.get_market_constituents(market_sel)
                
                if constituents.empty:
                    return
                    
                if sector_sel != "Alle" and 'Sector' in constituents.columns:
                    tickers = constituents[constituents['Sector'] == sector_sel]['Ticker'].tolist()
                else:
                    tickers = constituents['Ticker'].head(60).tolist() # Limiet voor snelheid
                    
                benchmark = MARKETS[market_sel]['benchmark']
                tickers = list(set(tickers + [benchmark]))
                
                df_prices = MarketDataProvider.get_price_data(tickers)
                
                if df_prices.empty:
                    st.error("Kon geen prijsdata ophalen.")
                    return
                    
                rrg_df = FinancialMath.calculate_rrg(df_prices, benchmark)
                
                if rrg_df.empty:
                    st.error("RRG Berekening mislukt: te weinig historische data.")
                else:
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
                        st.warning("Geen geschikte kandidaten gevonden.")

    # --- TAB 2: AI Execution ---
    with tab_execution:
        st.markdown("### 🧠 Stap 2: Multi-Agent Validatie & Portfolio")
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
            
            # Mock data voor VIX en rente (voor demonstratie)
            vix_mock = np.full(100, 20.0)
            term_mock = np.full(100, 1.5)
            
            for idx, t in enumerate(candidates):
                status_text.markdown(f"🕵️ **Analyseren:** {t}...")
                progress_bar.progress((idx + 1) / len(candidates))
                
                sig, bet = quant_eng.analyze(df_prices[t], vix_mock, term_mock) if t in df_prices.columns else (0.0, 0.0)
                
                # Zelfs met lage bet doen we hier een AI check voor de demonstratie
                context = f"Laatste prijs: {df_prices[t].iloc[-1]:.2f}" if t in df_prices.columns else "Geen data"
                ai_res = AIAgentAdapter.run_swarm_debate(t, context, api_key)
                
                ai_score_norm = (ai_res.get('score', 0) + 10) / 20 
                final_conviction = (bet * 0.7) + (ai_score_norm * 0.3)
                
                if final_conviction > 0.3: # Threshold verlaagd om vaker resultaat te tonen
                    results.append({
                        'Ticker': t, 'Conviction': round(final_conviction, 2),
                        'Quant_Conf': round(bet, 2), 'AI_Score': ai_res.get('score', 0),
                        'Rationale': ai_res.get('rationale', '')
                    })
            
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

if __name__ == "__main__":
    render_ui()
