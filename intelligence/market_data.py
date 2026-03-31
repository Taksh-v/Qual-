import logging
from dataclasses import dataclass
from typing import Optional

try:
    import yfinance as yf
    import pandas as pd
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MarketDataSnapshot:
    sp500: str
    ten_year: str
    vix: str
    dxy: str
    wti: str

    def format_for_prompt(self) -> str:
        return f"S&P500={self.sp500}, 10Y={self.ten_year}, VIX={self.vix}, DXY={self.dxy}, WTI={self.wti}"


def get_live_market_snapshot() -> MarketDataSnapshot:
    """Fetch live data from yfinance for key macro indicators."""
    fallback = MarketDataSnapshot("N/A", "N/A", "N/A", "N/A", "N/A")
    if not YFINANCE_AVAILABLE:
        logger.warning("yfinance not installed. Using fallback market data.")
        return fallback

    try:
        # Tickers:
        # ^GSPC: S&P 500
        # ^TNX: 10-Year Treasury Yield
        # ^VIX: CBOE Volatility Index
        # DX-Y.NYB: US Dollar Index
        # CL=F: Crude Oil WTI
        tickers = ['^GSPC', '^TNX', '^VIX', 'DX-Y.NYB', 'CL=F']
        
        # Download data for the last 5 days to ensure we get a valid close (e.g. over weekends)
        df = yf.download(tickers, period="5d", progress=False)
        
        if df.empty or 'Close' not in df.columns:
            logger.warning("yfinance returned empty data.")
            return fallback
            
        # Get the 'Close' dataframe
        close_df = df['Close']
        
        # Extract the last valid non-NaN value for each column
        last_values = {}
        for ticker in tickers:
            if ticker in close_df.columns:
                series = close_df[ticker].dropna()
                val = series.iloc[-1] if not series.empty else None
                last_values[ticker] = val
            else:
                last_values[ticker] = None
                
        def format_val(val: Optional[float], fmt: str) -> str:
            if val is None or pd.isna(val):
                return "N/A"
            return fmt.format(val)

        return MarketDataSnapshot(
            sp500=format_val(last_values['^GSPC'], "{:.0f}"),
            ten_year=format_val(last_values['^TNX'], "{:.2f}%"),
            vix=format_val(last_values['^VIX'], "{:.2f}"),
            dxy=format_val(last_values['DX-Y.NYB'], "{:.2f}"),
            wti=format_val(last_values['CL=F'], "${:.2f}")
        )
        
    except Exception as e:
        logger.error(f"Failed to fetch market data: {e}")
        return fallback
