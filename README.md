# Options Trading Simulator

A self-contained Python trading simulator with Streamlit UI that lets you trade options with news-driven price shocks.

## Features

- 📊 **10 Pre-loaded tickers** with customizable sectors, tags, prices, and IVs
- 📰 **Breaking news engine** with automatic sector mapping and price shocks
- 📈 **Options chain generator** with Black-Scholes pricing and Greeks
- 🎫 **Order system** with ATM/ITM/OTM presets
- 💼 **Position tracking** with real-time PnL and Greeks
- 📥 **CSV export** for positions and order history

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

## Quick Start

1. **Enter breaking news** - Type a headline and set the impact level
   - Example: "Tariffs increased on Chinese goods" → affects hardware/manufacturing/retail negatively, streaming positively

2. **View options chain** - Select a ticker to see strikes for 7/30/90 day expiries
   - Filters available by type (Call/Put) and moneyness (ITM/ATM/OTM)

3. **Place an order** - Choose ATM/ITM/OTM preset or select from chain
   - View live quote with Greeks
   - Monitor positions in real-time

4. **Track positions** - Automatic PnL updates as market moves
   - Export positions and order history to CSV

## Configuration

Edit `pyproject.toml` to adjust basedpyright/pyright type checking settings.

## Requirements

- Python 3.10+
- streamlit
- pandas
- numpy
- scipy

## Example Use Cases

### Testing News Impact
1. Start with default tickers
2. Enter "Oil prices surge" → see XOM (Energy) rise, others fall
3. Enter "Tariffs increased" → see hardware/manufacturing fall, SPOT/NFLX rise

### Options Trading
1. Select AAPL from chain
2. Choose ATM Call, 30 DTE
3. Place order and monitor PnL
4. Apply a news headline and watch position value change

