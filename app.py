"""
Self-contained Python trading simulator with Streamlit UI
Allows trading options with news-driven price/IV shocks

Run steps:
# pip install streamlit pandas numpy scipy
# streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, date, timedelta
import math
from typing import Optional, Dict, List, Tuple


def init_state():
    """Initialize session state with seed data"""
    if 'initialized' not in st.session_state:
        # Seed 10 tickers with sectors, tags, starting prices, IVs
        st.session_state.tickers = {
            'AAPL': {'sector': 'Hardware', 'tags': {'hardware', 'manufacturing'}, 'price': 180.0, 'iv': 0.35},
            'MSFT': {'sector': 'Software', 'tags': {'pure_software', 'cloud'}, 'price': 370.0, 'iv': 0.30},
            'AMZN': {'sector': 'Retail/Cloud', 'tags': {'retail', 'cloud', 'manufacturing'}, 'price': 145.0, 'iv': 0.35},
            'GOOGL': {'sector': 'Ads/Cloud', 'tags': {'ads', 'cloud'}, 'price': 140.0, 'iv': 0.30},
            'META': {'sector': 'Ads/Social', 'tags': {'ads', 'social'}, 'price': 380.0, 'iv': 0.40},
            'TSLA': {'sector': 'Auto/Manufacturing', 'tags': {'auto', 'manufacturing'}, 'price': 250.0, 'iv': 0.50},
            'NVDA': {'sector': 'Semis', 'tags': {'semis', 'hardware'}, 'price': 450.0, 'iv': 0.45},
            'JPM': {'sector': 'Bank', 'tags': {'bank', 'financials'}, 'price': 155.0, 'iv': 0.25},
            'XOM': {'sector': 'Energy', 'tags': {'energy', 'oil'}, 'price': 110.0, 'iv': 0.30},
            'NFLX': {'sector': 'Streaming', 'tags': {'streaming', 'pure_software'}, 'price': 450.0, 'iv': 0.35},
            'SPOT': {'sector': 'Streaming', 'tags': {'streaming', 'pure_software'}, 'price': 150.0, 'iv': 0.40},
        }
        
        # Session parameters
        st.session_state.r = 0.04  # risk-free rate
        st.session_state.default_iv = 0.35
        
        # Starting snapshot for % change calculation
        st.session_state.session_start_prices = {t: data['price'] for t, data in st.session_state.tickers.items()}
        
        # News rules - keyword detection and tag betas
        st.session_state.news_rules = {
            'keyword_map': {
                'tariff': ['tariff', 'tariffs', 'duties', 'duty', 'trade war', 'trade war'],
                'oil_spike': ['oil', 'brent', 'wti', 'crude', 'petroleum'],
                'rate_hike': ['rate hike', 'fed hike', 'tightening', 'hawkish', 'rates'],
                'earnings_beat': ['earnings beat', 'earnings surprise', 'exceeded expectations'],
                'scandal': ['scandal', 'fraud', 'investigation', 'lawsuit', 'violations'],
            },
            'tag_betas': {
                'tariff': {
                    'negative': {'hardware': -1.0, 'manufacturing': -1.0, 'retail': -0.7, 'semis': -0.6},
                    'positive': {'streaming': 0.5, 'pure_software': 0.3, 'cloud': 0.2}
                },
                'oil_spike': {
                    'positive': {'energy': 1.0, 'oil': 1.0},
                    'negative': {'airlines': -0.9, 'transport': -0.6, 'retail': -0.3, 'manufacturing': -0.4}
                },
                'rate_hike': {
                    'positive': {'bank': 0.4, 'financials': 0.3},
                    'negative': {'tech_growth': -0.8, 'real_estate': -0.6, 'growth': -0.5}
                },
                'earnings_beat': {
                    'positive': {'cloud': 0.5, 'semis': 0.6, 'streaming': 0.4},
                    'negative': {}
                },
                'scandal': {
                    'negative': {'ads': -0.8, 'social': -0.7, 'retail': -0.5},
                    'positive': {}
                }
            }
        }
        
        # News feed
        st.session_state.news_feed = []
        
        # Order book and positions
        st.session_state.order_book = []
        st.session_state.positions = {}  # key: (ticker, type, K, expiry_date) -> {qty, avg_cost}
        
        # UI state
        st.session_state.selected_ticker = 'AAPL'
        st.session_state.initialized = True


def detect_category(headline: str) -> Optional[str]:
    """Detect news category from headline text"""
    headline_lower = headline.lower()
    for category, keywords in st.session_state.news_rules['keyword_map'].items():
        if any(keyword in headline_lower for keyword in keywords):
            return category
    return None


def apply_headline(headline: str, category: Optional[str], chosen_tickers: List[str], impact: float):
    """Apply news headline to ticker prices and IVs"""
    timestamp = datetime.now().isoformat()
    
    affected_tickers = {}
    
    if category and category in st.session_state.news_rules['tag_betas']:
        betas = st.session_state.news_rules['tag_betas'][category]
        
        for ticker, data in st.session_state.tickers.items():
            tags = data['tags']
            
            # Check negative impacts
            for neg_tag, beta in betas.get('negative', {}).items():
                if neg_tag in tags:
                    if ticker not in affected_tickers:
                        affected_tickers[ticker] = 0.0
                    affected_tickers[ticker] += beta * impact
            
            # Check positive impacts
            for pos_tag, beta in betas.get('positive', {}).items():
                if pos_tag in tags:
                    if ticker not in affected_tickers:
                        affected_tickers[ticker] = 0.0
                    affected_tickers[ticker] += beta * impact
    
    # Apply manual selections
    for ticker in chosen_tickers:
        if ticker in st.session_state.tickers:
            affected_tickers[ticker] = impact
    
    # Apply shocks
    for ticker, beta in affected_tickers.items():
        if ticker not in st.session_state.tickers:
            continue
            
        data = st.session_state.tickers[ticker]
        old_price = data['price']
        
        # Price shock: 0.25% base * impact * beta
        base_move = 0.25 * impact / 100  # Convert to decimal
        price_move = base_move * beta if beta != 0 else base_move if ticker in chosen_tickers else 0
        new_price = old_price * (1 + price_move)
        new_price = max(0.01, new_price)  # Clamp minimum
        
        # IV shock
        sigma_old = data['iv']
        iv_boost = 0.02 * abs(impact) * (1 + max(0, -np.sign(impact * beta) * 0.5)) if beta != 0 else 0.02 * abs(impact)
        new_sigma = sigma_old + iv_boost
        new_sigma = max(0.10, min(1.00, new_sigma))  # Clamp [0.10, 1.00]
        
        st.session_state.tickers[ticker]['price'] = new_price
        st.session_state.tickers[ticker]['iv'] = new_sigma
    
    # Log to news feed
    st.session_state.news_feed.append({
        'timestamp': timestamp,
        'headline': headline,
        'category': category or 'uncategorized',
        'impact': impact,
        'tickers': list(affected_tickers.keys()) if affected_tickers else []
    })


def strike_grid(S: float) -> np.ndarray:
    """Generate strike grid around spot price"""
    if S < 50:
        step = 1.0
    elif S < 200:
        step = 2.5
    else:
        step = 5.0
    
    # Range: -20% to +20%
    min_strike = S * 0.8
    max_strike = S * 1.2
    
    # Round to step
    min_strike = math.floor(min_strike / step) * step
    max_strike = math.ceil(max_strike / step) * step
    
    strikes = np.arange(min_strike, max_strike + step, step)
    return strikes


def bs_price_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict:
    """Black-Scholes price and Greeks for European options"""
    if T <= 0:
        # Expired option
        intrinsic = max(S - K, 0) if option_type == 'CALL' else max(K - S, 0)
        return {
            'price': intrinsic,
            'delta': 1.0 if option_type == 'CALL' and S > K else -1.0 if option_type == 'PUT' and S < K else 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0
        }
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'CALL':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:  # PUT
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
        theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    return {
        'price': max(0.0, price),
        'delta': delta,
        'gamma': gamma / 100,  # Scale gamma
        'theta': theta / 365,  # Per day
        'vega': vega / 100  # Per 1% vol change
    }


def preset_strike(S: float, option_type: str, preset: str) -> float:
    """Calculate strike for ATM/ITM/OTM presets"""
    if preset == 'ATM':
        # Nearest strike to spot
        return S
    
    # Determine strike step
    if S < 50:
        step = 1.0
    elif S < 200:
        step = 2.5
    else:
        step = 5.0
    
    if option_type == 'CALL':
        if preset == 'ITM':
            target = S * 0.95  # Below spot for ITM calls
        else:  # OTM
            target = S * 1.05  # Above spot for OTM calls
    else:  # PUT
        if preset == 'ITM':
            target = S * 1.05  # Above spot for ITM puts
        else:  # OTM
            target = S * 0.95  # Below spot for OTM puts
    
    # Round to nearest strike
    return round(target / step) * step


def build_chain(ticker: str, expiries: List[int]) -> pd.DataFrame:
    """Build options chain for ticker with multiple expiries"""
    if ticker not in st.session_state.tickers:
        return pd.DataFrame()
    
    data = st.session_state.tickers[ticker]
    S = data['price']
    sigma = data['iv']
    r = st.session_state.r
    
    strikes = strike_grid(S)
    now = datetime.now()
    
    chain_rows = []
    for expiry_days in expiries:
        expiry_date = now + timedelta(days=expiry_days)
        T = expiry_days / 365.25
        
        for K in strikes:
            # Calls
            call_res = bs_price_greeks(S, K, T, r, sigma, 'CALL')
            chain_rows.append({
                'DTE': expiry_days,
                'Expiry': expiry_date.strftime('%Y-%m-%d'),
                'Strike': K,
                'Type': 'CALL',
                'Price': call_res['price'],
                'Delta': call_res['delta'],
                'Gamma': call_res['gamma'],
                'Theta': call_res['theta'],
                'Vega': call_res['vega'],
                'Moneyness': 'ITM' if S > K else 'ATM' if abs(S - K) < 0.01 else 'OTM'
            })
            
            # Puts
            put_res = bs_price_greeks(S, K, T, r, sigma, 'PUT')
            chain_rows.append({
                'DTE': expiry_days,
                'Expiry': expiry_date.strftime('%Y-%m-%d'),
                'Strike': K,
                'Type': 'PUT',
                'Price': put_res['price'],
                'Delta': put_res['delta'],
                'Gamma': put_res['gamma'],
                'Theta': put_res['theta'],
                'Vega': put_res['vega'],
                'Moneyness': 'ITM' if S < K else 'ATM' if abs(S - K) < 0.01 else 'OTM'
            })
    
    return pd.DataFrame(chain_rows)


def place_order(ticker: str, option_type: str, preset: str, K: Optional[float], expiry_days: int, qty: int):
    """Place an options order"""
    if ticker not in st.session_state.tickers:
        return False
    
    data = st.session_state.tickers[ticker]
    S = data['price']
    sigma = data['iv']
    r = st.session_state.r
    
    # Determine strike
    if K is None:
        K = preset_strike(S, option_type, preset)
    
    # Calculate expiry date
    expiry_date = datetime.now() + timedelta(days=expiry_days)
    
    # Get option price
    T = expiry_days / 365.25
    greeks = bs_price_greeks(S, K, T, r, sigma, option_type)
    fill_price = greeks['price']
    
    # Create order record
    order = {
        'timestamp': datetime.now().isoformat(),
        'ticker': ticker,
        'type': option_type,
        'preset': preset,
        'K': K,
        'expiry_date': expiry_date.strftime('%Y-%m-%d'),
        'DTE': expiry_days,
        'qty': qty,
        'fill_price': fill_price,
        'S_at_fill': S,
        'IV_at_fill': sigma
    }
    
    st.session_state.order_book.append(order)
    
    # Update positions
    position_key = (ticker, option_type, K, expiry_date.strftime('%Y-%m-%d'))
    if position_key in st.session_state.positions:
        pos = st.session_state.positions[position_key]
        total_cost = pos['avg_cost'] * pos['qty'] + fill_price * qty
        pos['qty'] += qty
        pos['avg_cost'] = total_cost / pos['qty']
    else:
        st.session_state.positions[position_key] = {'qty': qty, 'avg_cost': fill_price}
    
    return True


def revalue_positions() -> pd.DataFrame:
    """Calculate current values and PnL for all positions"""
    if not st.session_state.positions:
        return pd.DataFrame()
    
    now = datetime.now()
    position_rows = []
    
    for (ticker, option_type, K, expiry_date_str), pos_data in st.session_state.positions.items():
        if ticker not in st.session_state.tickers:
            continue
        
        # Parse expiry date
        try:
            expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d')
            DTE = (expiry_date - now).days
            T = max(DTE / 365.25, 0)
        except:
            continue
        
        # Current market data
        ticker_data = st.session_state.tickers[ticker]
        S = ticker_data['price']
        sigma = ticker_data['iv']
        r = st.session_state.r
        
        # Current mark
        greeks = bs_price_greeks(S, K, T, r, sigma, option_type)
        mark = greeks['price']
        
        # PnL
        unrealized_pnl = (mark - pos_data['avg_cost']) * pos_data['qty']
        
        position_rows.append({
            'Ticker': ticker,
            'Type': option_type,
            'Strike': K,
            'Expiry': expiry_date_str,
            'DTE': max(DTE, 0),
            'Qty': pos_data['qty'],
            'Avg Cost': round(pos_data['avg_cost'], 2),
            'Mark': round(mark, 2),
            'Unrealized PnL': round(unrealized_pnl, 2),
            'Delta': round(greeks['delta'], 3),
            'Theta/day': round(greeks['theta'], 3),
            'Vega': round(greeks['vega'], 3)
        })
    
    return pd.DataFrame(position_rows)


def render_header():
    """Render main header"""
    st.title("📈 Options Trading Simulator")
    st.markdown("Simulate options trading with news-driven price shocks")


def render_sidebar():
    """Render sidebar controls"""
    with st.sidebar:
        st.header("⚙️ Session Settings")
        
        # Global parameters
        r = st.number_input("Risk-Free Rate (r)", min_value=0.0, max_value=0.10, value=st.session_state.r, step=0.001, format="%.3f")
        st.session_state.r = r
        
        default_iv = st.number_input("Default IV", min_value=0.10, max_value=1.00, value=st.session_state.default_iv, step=0.01, format="%.2f")
        st.session_state.default_iv = default_iv
        
        if st.button("🔄 Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.initialized = False
        
        st.divider()
        
        # Starting data editor
        st.header("📊 Starting Data")
        st.markdown("Edit ticker data below")
        
        ticker_editor_rows = []
        for ticker, data in st.session_state.tickers.items():
            ticker_editor_rows.append({
                'Ticker': ticker,
                'Sector': data['sector'],
                'Tags': ', '.join(data['tags']),
                'Price': data['price'],
                'IV': data['iv']
            })
        
        df_editor = pd.DataFrame(ticker_editor_rows)
        edited_df = st.data_editor(df_editor, num_rows="dynamic", key="ticker_editor")
        
        if st.button("💾 Apply Changes"):
            for idx, row in edited_df.iterrows():
                ticker = row['Ticker']
                if ticker in st.session_state.tickers:
                    st.session_state.tickers[ticker]['price'] = float(row['Price'])
                    st.session_state.tickers[ticker]['iv'] = float(row['IV'])
                    st.session_state.tickers[ticker]['sector'] = row['Sector']
                    # Parse tags
                    tags = {t.strip() for t in row['Tags'].split(',') if t.strip()}
                    st.session_state.tickers[ticker]['tags'] = tags
            st.success("Changes applied!")
            st.rerun()


def render_news_section():
    """Render news headline input and market table"""
    st.header("📰 Breaking News")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        headline = st.text_input("Headline", placeholder="e.g., Tariffs increased on Chinese goods")
        
        # Auto-detect category
        category = detect_category(headline)
        detected_cat = st.text_input("Category (auto-detected)", value=category or "", disabled=True)
    
    with col2:
        # Available tickers
        available_tickers = list(st.session_state.tickers.keys())
        chosen_tickers = st.multiselect("Affected Tickers", available_tickers)
        
        impact = st.slider("Impact", min_value=-3.0, max_value=3.0, value=0.0, step=0.1, help="Negative = bad news, Positive = good news")
    
    if st.button("📰 Apply Headline", type="primary"):
        if headline:
            apply_headline(headline, category, chosen_tickers, impact)
            st.success("Headline applied!")
            st.rerun()
    
    st.divider()
    
    # Live market table
    st.subheader("Live Market Data")
    market_rows = []
    for ticker, data in st.session_state.tickers.items():
        session_start = st.session_state.session_start_prices.get(ticker, data['price'])
        pct_change = ((data['price'] - session_start) / session_start * 100) if session_start > 0 else 0
        
        market_rows.append({
            'Ticker': ticker,
            'Sector/Tags': f"{data['sector']} ({', '.join(list(data['tags'])[:3])})",
            'Price': f"${data['price']:.2f}",
            'IV': f"{data['iv']:.2f}",
            'Day Δ%': f"{pct_change:+.2f}%"
        })
    
    df_market = pd.DataFrame(market_rows)
    st.dataframe(df_market, use_container_width=True, hide_index=True)


def render_manual_price_editor():
    """Render manual price editor"""
    with st.expander("🔧 Manual Price Editor"):
        st.markdown("Override prices manually")
        
        edit_rows = []
        for ticker, data in st.session_state.tickers.items():
            edit_rows.append({
                'Ticker': ticker,
                'Price': data['price'],
                'IV': data['iv']
            })
        
        df_edit = pd.DataFrame(edit_rows)
        edited = st.data_editor(df_edit, key="manual_price_editor")
        
        if st.button("📊 Update Prices"):
            for idx, row in edited.iterrows():
                ticker = row['Ticker']
                st.session_state.tickers[ticker]['price'] = float(row['Price'])
                st.session_state.tickers[ticker]['iv'] = float(row['IV'])
            st.rerun()


def render_options_chain():
    """Render options chain and order ticket"""
    st.header("📊 Options Chain")
    
    # Ticker selector
    available_tickers = list(st.session_state.tickers.keys())
    selected_ticker = st.selectbox("Select Ticker", available_tickers, key="chain_ticker")
    
    if not selected_ticker:
        return
    
    # Build chain
    chain_df = build_chain(selected_ticker, [7, 30, 90])
    
    if chain_df.empty:
        st.info("No chain data")
        return
    
    # Filter options
    st.subheader("Chain Data")
    
    col1, col2 = st.columns(2)
    with col1:
        filter_type = st.selectbox("Filter by Type", ['All', 'Call', 'Put'], key="chain_type")
    with col2:
        filter_moneyness = st.selectbox("Filter by Moneyness", ['All', 'ITM', 'ATM', 'OTM'], key="chain_moneyness")
    
    # Apply filters
    filtered_chain = chain_df.copy()
    if filter_type != 'All':
        filtered_chain = filtered_chain[filtered_chain['Type'] == filter_type.upper()] # type: ignore
    if filter_moneyness != 'All':
        filtered_chain = filtered_chain[filtered_chain['Moneyness'] == filter_moneyness]
    
    # Display chain
    display_chain = filtered_chain[['Type', 'Strike', 'Expiry', 'DTE', 'Price', 'Delta', 'Theta', 'Vega']].copy()
    st.dataframe(display_chain, use_container_width=True, hide_index=True)
    
    return selected_ticker, filtered_chain


def render_order_ticket():
    """Render order ticket UI"""
    st.header("🎫 Order Ticket")
    
    available_tickers = list(st.session_state.tickers.keys())
    ticket_ticker = st.selectbox("Ticker", available_tickers, key="order_ticker")
    
    col1, col2 = st.columns(2)
    
    with col1:
        option_type = st.radio("Type", ['CALL', 'PUT'], horizontal=True)
        preset = st.radio("Moneyness Preset", ['ATM', 'ITM', 'OTM'], horizontal=True)
        qty = st.number_input("Quantity", min_value=1, value=1, step=1)
    
    with col2:
        expiry_days = st.selectbox("Expiry", [7, 30, 90], format_func=lambda x: f"{x} days")
    
    # Show quote preview
    if ticket_ticker and ticket_ticker in st.session_state.tickers:
        data = st.session_state.tickers[ticket_ticker]
        S = data['price']
        sigma = data['iv']
        r = st.session_state.r
        
        K = preset_strike(S, option_type, preset) # type: ignore
        T = expiry_days / 365.25 # type: ignore
        
        greeks = bs_price_greeks(S, K, T, r, sigma, option_type) # type: ignore
        
        st.markdown("**Quote Preview**")
        quote_cols = st.columns(2)
        with quote_cols[0]:
            st.metric("Spot", f"${S:.2f}")
            st.metric("Strike", f"${K:.2f}")
            st.metric("DTE", f"{expiry_days} days")
        with quote_cols[1]:
            st.metric("Theo Price", f"${greeks['price']:.2f}")
            st.metric("Delta", f"{greeks['delta']:.3f}")
            st.metric("Theta/day", f"{greeks['theta']:.3f}")
            st.metric("Vega", f"{greeks['vega']:.3f}")
    
    if st.button("🛒 Place Order", type="primary"):
        if place_order(ticket_ticker, option_type, preset, None, expiry_days, qty): # type: ignore
            st.success("Order placed successfully!")
            st.rerun()
        else:
            st.error("Order failed!")


def render_positions_section():
    """Render positions and order book"""
    st.header("💼 Positions & Orders")
    
    # Positions
    positions_df = revalue_positions()
    
    if not positions_df.empty:
        st.subheader("Current Positions")
        st.dataframe(positions_df, use_container_width=True, hide_index=True)
        
        # Totals row
        if len(positions_df) > 0:
            st.markdown("**Totals**")
            st.metric("Total Contracts", int(positions_df['Qty'].sum()))
            st.metric("Total Unrealized PnL", f"${positions_df['Unrealized PnL'].sum():.2f}")
    else:
        st.info("No open positions")
    
    st.divider()
    
    # Order book
    st.subheader("Recent Orders (Last 20)")
    if st.session_state.order_book:
        order_df = pd.DataFrame(st.session_state.order_book[-20:])
        if not order_df.empty:
            display_orders = order_df[['timestamp', 'ticker', 'type', 'K', 'expiry_date', 'DTE', 'qty', 'fill_price']]
            st.dataframe(display_orders, use_container_width=True, hide_index=True)
    else:
        st.info("No orders yet")
    
    st.divider()
    
    # Export buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export Positions CSV"):
            if not positions_df.empty:
                csv = positions_df.to_csv(index=False)
                st.download_button("Download Positions", csv, "positions.csv", "text/csv")
    with col2:
        if st.button("📥 Export Order Book CSV"):
            if st.session_state.order_book:
                order_df = pd.DataFrame(st.session_state.order_book)
                csv = order_df.to_csv(index=False)
                st.download_button("Download Order Book", csv, "order_book.csv", "text/csv")


def main():
    """Main app entry point"""
    init_state()
    
    render_header()
    
    # Layout: sidebar + main
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # News and prices
        render_news_section()
        render_manual_price_editor()
        
        st.divider()
        
        # Options chain
        render_options_chain()
        
        st.divider()
        
        # Order ticket
        render_order_ticket()
    
    with col2:
        render_positions_section()
    
    # Sidebar
    render_sidebar()


if __name__ == "__main__":
    main()

