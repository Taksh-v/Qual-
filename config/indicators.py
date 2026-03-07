"""
config/indicators.py
---------------------
Central registry of all indicator keys, display labels, and units.
Used by the API layer, dashboard, and export functions.
"""
from __future__ import annotations

# Format: key -> (display_label, unit_string)
INDICATOR_META: dict[str, tuple[str, str]] = {
    # ── Policy & short rates ──────────────────────────────────────────────
    "fed_funds_rate":       ("Fed Funds",           "%"),
    "yield_3m":             ("3M Yield",            "%"),
    "yield_1y":             ("1Y Yield",            "%"),
    # ── Treasury yield curve ──────────────────────────────────────────────
    "yield_2y":             ("2Y Yield",            "%"),
    "yield_10y":            ("10Y Yield",           "%"),
    "yield_30y":            ("30Y Yield",           "%"),
    "yield_curve":          ("Curve 10Y-2Y",        "bps"),
    "yield_curve_10y3m":    ("Curve 10Y-3M",        "bps"),
    "yield_curve_30y2y":    ("Curve 30Y-2Y",        "bps"),
    "term_premium_proxy":   ("Term Premium",        "bps"),
    # ── Real rates ────────────────────────────────────────────────────────
    "real_rate_proxy":      ("Real Rate (10Y-CPI)", "%"),
    "real_rate_10y":        ("10Y TIPS",            "%"),
    "real_rate_5y":         ("5Y TIPS",             "%"),
    "fed_real_rate":        ("Fed Real Rate",       "%"),
    # ── Inflation ─────────────────────────────────────────────────────────
    "inflation_cpi":        ("CPI",                 "%"),
    "inflation_core_cpi":   ("Core CPI",            "%"),
    "pce_deflator":         ("PCE Deflator",        "%"),
    "pce_core":             ("Core PCE",            "%"),
    "breakeven_5y":         ("5Y Breakeven",        "%"),
    "breakeven_10y":        ("10Y Breakeven",       "%"),
    "breakeven_5y10y_slope":("BE Slope 5-10Y",     "%"),
    # ── Growth & activity ─────────────────────────────────────────────────
    "gdp_growth":           ("GDP Growth",          "%"),
    "us_industrial_prod":   ("Industrial Prod",     "idx"),
    "capacity_utilization": ("Capacity Util",       "%"),
    "us_retail_sales":      ("Retail Sales",        "$bn"),
    "us_housing_starts":    ("Housing Starts",      "k"),
    "us_building_permits":  ("Building Permits",    "k"),
    "us_trade_balance":     ("Trade Balance",       "$bn"),
    "conf_board_lei":       ("Lead Econ Idx",       "idx"),
    # ── Labour ────────────────────────────────────────────────────────────
    "unemployment":         ("Unemployment",        "%"),
    "initial_claims":       ("Initial Claims",      "k"),
    "continued_claims":     ("Cont. Claims",        "k"),
    "jolts_openings":       ("JOLTS Openings",      "mn"),
    "nonfarm_payrolls":     ("Nonfarm Payrolls",    "k"),
    "participation_rate":   ("Labour Particip",     "%"),
    # ── Surveys & sentiment ───────────────────────────────────────────────
    "pmi_mfg":              ("PMI Mfg",             ""),
    "consumer_sentiment":   ("Consumer Sentiment",  ""),
    # ── Credit ────────────────────────────────────────────────────────────
    "credit_hy":            ("HY Spread",           "bps"),
    "credit_ig":            ("IG Spread",           "bps"),
    "credit_bb":            ("BB Spread",           "bps"),
    "credit_spread_gap":    ("HY-IG Gap",           "bps"),
    "mort_rate_30y":        ("30Y Mortgage",        "%"),
    "ted_spread":           ("TED Spread",          "bps"),
    # ── Money ─────────────────────────────────────────────────────────────
    "m2_money_supply":      ("M2",                  "$bn"),
    "fed_balance_sheet":    ("Fed Balance Sheet",   "$bn"),
    "m2_velocity":          ("M2 Velocity",         ""),
    # ── Equities ──────────────────────────────────────────────────────────
    "sp500":                ("S&P 500",             ""),
    "nasdaq":               ("Nasdaq",              ""),
    "vix":                  ("VIX",                 ""),
    "dow":                  ("Dow Jones",           ""),
    "russell2000":          ("Russell 2000",        ""),
    "sector_tech":          ("XLK Tech",            ""),
    "sector_energy":        ("XLE Energy",          ""),
    "sector_finance":       ("XLF Finance",         ""),
    "sector_health":        ("XLV Health",          ""),
    "sector_consumer":      ("XLY Consumer",        ""),
    # ── India ─────────────────────────────────────────────────────────────
    "nifty50":              ("Nifty 50",            ""),
    "sensex":               ("Sensex",              ""),
    "nifty_bank":           ("Nifty Bank",          ""),
    "nifty_it":             ("Nifty IT",            ""),
    "usd_inr":              ("USD/INR",             ""),
    "india_gdp_growth":     ("India GDP",           "%"),
    "india_inflation_cpi":  ("India CPI",           "%"),
    "india_current_account":("India C/A",           "% GDP"),
    # ── FX ────────────────────────────────────────────────────────────────
    "dxy":                  ("DXY",                 ""),
    "eur_usd":              ("EUR/USD",             ""),
    "gbp_usd":              ("GBP/USD",             ""),
    "usd_jpy":              ("USD/JPY",             ""),
    "usd_cny":              ("USD/CNY",             ""),
    # ── Commodities ───────────────────────────────────────────────────────
    "gold":                 ("Gold",                "$"),
    "silver":               ("Silver",              "$"),
    "oil_wti":              ("WTI Oil",             "$"),
    "oil_brent":            ("Brent Oil",           "$"),
    "natural_gas":          ("Nat Gas",             "$"),
    "copper":               ("Copper",              "$"),
    # ── Crypto ────────────────────────────────────────────────────────────
    "btc_usd":              ("Bitcoin",             "$"),
    "eth_usd":              ("Ethereum",            "$"),
    # ── Global indices ────────────────────────────────────────────────────
    "ftse100":              ("FTSE 100",            ""),
    "nikkei225":            ("Nikkei 225",          ""),
    "hangseng":             ("Hang Seng",           ""),
    "dax":                  ("DAX",                 ""),
}
