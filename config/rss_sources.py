"""
rss_sources.py
--------------
Curated RSS / Atom feed catalogue for broad financial market coverage.
Organised by category so the ingestor can fetch selectively or all at once.

All feeds are free / public.  No API key required.
"""

# Each entry: (label, feed_url)
# label is used as the 'source' tag in chunk metadata.

RSS_FEEDS: dict[str, list[tuple[str, str]]] = {

    # ── US Macro & Fed ────────────────────────────────────────────────────────
    "us_macro": [
        ("Bloomberg Economics",     "https://feeds.bloomberg.com/economics/news.rss"),
        ("Reuters Business",        "https://feeds.reuters.com/reuters/businessNews"),
        ("Reuters Markets",         "https://feeds.reuters.com/reuters/UKBusiness"),
        ("CNBC Economy",            "https://www.cnbc.com/id/20910258/device/rss/rss.html"),
        ("WSJ Economy",             "https://feeds.content.dowjones.io/public/rss/mw_realestate"),
        ("MarketWatch Economy",     "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
        ("FT Markets",              "https://www.ft.com/rss/markets"),
        ("FT Economics",            "https://www.ft.com/rss/economics"),
        ("Calculated Risk",         "https://www.calculatedriskblog.com/feeds/posts/default"),
        ("Fed Reserve Atlanta",     "https://www.frbatlanta.org/rss/news"),
        ("Fed Reserve NY",          "https://feeds.newyorkfed.org/medialibrary/media/research/staff_reports/sr.xml"),
    ],

    # ── Global Central Banks ──────────────────────────────────────────────────
    "central_banks": [
        ("BIS Press",               "https://www.bis.org/press.rss"),
        ("IMF News",                "https://www.imf.org/en/News/RSS"),
        ("World Bank News",         "https://feeds.worldbank.org/worldbank/news"),
        ("ECB Press",               "https://www.ecb.europa.eu/rss/press.html"),
        ("Bank of England",         "https://www.bankofengland.co.uk/rss/speeches"),
        ("RBI Notifications",       "https://www.rbi.org.in/rss/NotificationsView.aspx"),
        ("OECD Latest",             "https://www.oecd.org/newsroom/rss.xml"),
    ],

    # ── Equities & Earnings ───────────────────────────────────────────────────
    "equities": [
        ("CNBC Markets",            "https://www.cnbc.com/id/15839069/device/rss/rss.html"),
        ("Bloomberg Markets",       "https://feeds.bloomberg.com/markets/news.rss"),
        ("Seeking Alpha Markets",   "https://seekingalpha.com/feed.xml"),
        ("BusinessInsider Markets", "https://markets.businessinsider.com/rss/news"),
        ("Reuters Stocks",          "https://feeds.reuters.com/reuters/companyNews"),
        ("Yahoo Finance",           "https://finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US"),
        ("Investopedia",            "https://www.investopedia.com/feeds/news.xml"),
        ("Barron's",                "https://www.barrons.com/xml/rss/3_7514.xml"),
        ("Motley Fool",             "https://www.fool.com/feeds/index.aspx"),
    ],

    # ── Fixed Income & Credit ─────────────────────────────────────────────────
    "fixed_income": [
        ("Bond Buyer",              "https://www.bondbuyer.com/feed"),
        ("Bloomberg Rates",         "https://feeds.bloomberg.com/rates-bonds/news.rss"),
        ("Credit Suisse Research",  "https://www.credit-suisse.com/rss/researchPublications.xml"),
    ],

    # ── Commodities & Energy ──────────────────────────────────────────────────
    "commodities": [
        ("Reuters Commodities",     "https://feeds.reuters.com/reuters/commoditiesNews"),
        ("Platts Energy",           "https://www.spglobal.com/platts/en/rss-feed/oil"),
        ("OilPrice.com",            "https://oilprice.com/rss/main"),
        ("Gold Price",              "https://goldprice.org/rss/gold-news.xml"),
        ("CNBC Commodities",        "https://www.cnbc.com/id/15839064/device/rss/rss.html"),
        ("Bloomberg Energy",        "https://feeds.bloomberg.com/energy-and-oil/news.rss"),
        ("Natural Gas Intel",       "https://www.naturalgasintel.com/feed/"),
    ],

    # ── India Markets ─────────────────────────────────────────────────────────
    "india": [
        ("Economic Times Markets",  "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
        ("Economic Times Economy",  "https://economictimes.indiatimes.com/news/economy/rssfeeds/1386920271.cms"),
        ("Livemint Markets",        "https://www.livemint.com/rss/markets"),
        ("Livemint Economy",        "https://www.livemint.com/rss/economy"),
        ("Business Standard",       "https://www.business-standard.com/rss/markets-106.rss"),
        ("Hindu Business",          "https://www.thehindu.com/business/Economy/?service=rss"),
        ("MoneyControl News",       "https://www.moneycontrol.com/rss/marketsnews.xml"),
        ("MoneyControl Economy",    "https://www.moneycontrol.com/rss/economy.xml"),
        ("NDTV Business",           "https://feeds.feedburner.com/NdtvProfitLatestUpdates"),
        ("Financial Express",       "https://www.financialexpress.com/feed/"),
        ("RBI Monetary",            "https://www.rbi.org.in/rss/MonetaryPolicyView.aspx"),
        ("SEBI",                    "https://www.sebi.gov.in/rss/sebiNewsUpdates.xml"),
    ],

    # ── Geopolitics & Trade ───────────────────────────────────────────────────
    "geopolitics": [
        ("Reuters World",           "https://feeds.reuters.com/Reuters/worldNews"),
        ("Bloomberg Politics",      "https://feeds.bloomberg.com/politics/news.rss"),
        ("FT World",                "https://www.ft.com/rss/world"),
        ("Stratfor",                "https://worldview.stratfor.com/rss.xml"),
        ("Foreign Affairs",         "https://www.foreignaffairs.com/rss.xml"),
        ("VOA Economy",             "https://www.voanews.com/podcast/3160.xml"),
        ("Al Jazeera Economy",      "https://www.aljazeera.com/xml/rss/all.xml"),
        ("South China Morning Post","https://www.scmp.com/rss/91/feed"),
    ],

    # ── Tech & AI (market-moving) ─────────────────────────────────────────────
    "tech_ai": [
        ("TechCrunch",              "https://techcrunch.com/feed/"),
        ("The Verge Tech",          "https://www.theverge.com/rss/index.xml"),
        ("Bloomberg Tech",          "https://feeds.bloomberg.com/technology/news.rss"),
        ("MIT Technology Review",   "https://www.technologyreview.com/feed/"),
        ("Wired Business",          "https://www.wired.com/feed/business/rss"),
        ("Ars Technica",            "https://feeds.arstechnica.com/arstechnica/index"),
        ("CNBC Tech",               "https://www.cnbc.com/id/19854910/device/rss/rss.html"),
    ],

    # ── Alternative / Macro Research ─────────────────────────────────────────
    "macro_research": [
        ("Project Syndicate",       "https://www.project-syndicate.org/rss"),
        ("Harvard Business Review", "https://hbr.org/feed"),
        ("VoxEU",                   "https://voxeu.org/rss.xml"),
        ("Brookings",               "https://www.brookings.edu/feed/"),
        ("Peterson Institute",      "https://www.piie.com/rss.xml"),
        ("NBER Working Papers",     "https://www.nber.org/rss/new_working_papers.xml"),
        ("Econbrowser",             "https://econbrowser.com/feed"),
        ("Macro Musings",           "https://www.cato.org/multimedia/macro-musings/rss"),
    ],

    # ── Crypto & Digital Assets ───────────────────────────────────────────────
    "crypto": [
        ("CoinDesk",                "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("CoinTelegraph",           "https://cointelegraph.com/rss"),
        ("Decrypt",                 "https://decrypt.co/feed"),
        ("The Block",               "https://www.theblock.co/rss.xml"),
    ],
}

# Flattened list of all feeds (for full ingestion runs)
ALL_FEEDS: list[tuple[str, str, str]] = [
    (category, label, url)
    for category, feeds in RSS_FEEDS.items()
    for label, url in feeds
]

# High-priority feeds for quick refresh (most market-moving)
PRIORITY_FEEDS: list[tuple[str, str, str]] = [
    (cat, lbl, url)
    for cat, feeds in RSS_FEEDS.items()
    if cat in ("us_macro", "equities", "india", "commodities", "central_banks")
    for lbl, url in feeds
]
