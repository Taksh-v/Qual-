"""
sources.py — Curated seed article list for the RAG knowledge base.
─────────────────────────────────────────────────────────────────────
Each URL is a high-signal financial / macro article. This list is
intentionally lean and finance-focused; noisy content degrades RAG
answer quality. Continuous fresh content arrives via the RSS pipeline
(config/rss_sources.py). Add new articles here only if they provide
durable reference value (data releases, earnings, policy decisions).

Content filters applied:
  KEEP: markets, macro, earnings, central banks, trade, AI market-impact
        commodities, FX, structured products, regulatory filings
  REMOVE: sports, entertainment, politics (non-market), crime, lifestyle
"""

NEWS_SOURCES: list[str] = [

    # ── US Macro & Markets ─────────────────────────────────────────────────
    "https://www.cnbc.com/2026/02/03/ai-disruption-fears-rock-software-stocks-how-jim-cramer-navigating-sell-off.html",
    "https://www.cnbc.com/2026/02/03/jim-cramers-top-10-things-to-watch-in-the-stock-market-tuesday.html",
    "https://www.cnbc.com/2026/02/03/musk-xai-spacex-biggest-merger-ever.html",
    "https://www.cnbc.com/2026/02/03/softbank-subsidiary-saimemory-intel-ai-memory-z-angle-program.html",
    "https://www.cnbc.com/2026/02/04/amazon-alexa-plus-us-releas.html",
    "https://www.cnbc.com/2026/02/04/anthropic-no-ads-claude-chatbot-openai-chatgpt.html",
    "https://www.cnbc.com/2026/02/04/trump-india-deal-russia-oil-purchases-kremlin-reaction.html",
    "https://www.cnbc.com/2026/02/04/alphabet-googl-q4-2025-earnings.html",
    "https://www.cnbc.com/2026/02/04/chipotle-cmg-q4-2025-earnings.html",
    "https://www.cnbc.com/2026/02/02/nvidia-stock-price-openai-funding.html",

    # ── Bloomberg Markets & Economics ────────────────────────────────────────
    "https://www.bloomberg.com/news/articles/2026-02-04/us-stocks-dip-as-amd-results-ai-concerns-weigh-on-sentiment",
    "https://www.bloomberg.com/news/articles/2026-02-04/software-stocks-are-now-sentenced-before-trial-jpmorgan-says",
    "https://www.bloomberg.com/news/articles/2026-02-04/ai-disruption-concerns-sink-software-makers-stocks-in-asia",
    "https://www.bloomberg.com/news/articles/2026-02-04/ai-chip-startup-positron-raises-230-million-from-arm-qatar-to-compete-with-nvidia",
    "https://www.bloomberg.com/news/articles/2026-02-04/amazon-launches-ai-enhanced-alexa-for-prime-subscribers-in-us",
    "https://www.bloomberg.com/news/articles/2026-02-04/adobe-boosts-ad-spending-to-1-4-billion-to-attack-fear-over-ai",
    "https://www.bloomberg.com/news/articles/2026-02-04/cerebras-raises-1-billion-in-funding-at-23-billion-valuation",
    "https://www.bloomberg.com/news/articles/2026-02-04/xi-holds-phone-call-with-trump-xinhua-reports",
    "https://www.bloomberg.com/news/articles/2026-02-04/eu-nations-agree-on-framework-to-give-ukraine-90-billion-loan",
    "https://www.bloomberg.com/news/articles/2026-02-04/german-finance-chief-says-new-ai-center-will-bolster-sovereignty",
    "https://www.bloomberg.com/news/articles/2026-02-04/china-raises-serious-concerns-with-eu-wind-power-subsidy-probe",
    "https://www.bloomberg.com/news/articles/2026-02-04/infineon-forecasts-growing-revenue-from-ai-data-center-demand",
    "https://www.bloomberg.com/news/articles/2026-02-04/elon-musk-s-spacex-said-to-open-ipo-pitching-to-non-us-banks",
    "https://www.bloomberg.com/news/articles/2026-02-04/resolve-ai-hits-1-billion-valuation-for-outage-thwarting-ai-agents",
    "https://www.bloomberg.com/news/articles/2026-02-04/microsoft-s-deal-to-provide-computing-to-openai-raises-alarms",
    "https://www.bloomberg.com/news/articles/2026-02-03/us-stocks-teeter-on-edge-of-record-as-palantir-earnings-shine",
    "https://www.bloomberg.com/news/articles/2026-02-03/walmart-joins-1-trillion-club-as-tech-frugal-shoppers-fuel-gains",
    "https://www.bloomberg.com/news/articles/2026-02-03/private-equity-s-giant-software-bet-has-been-upended-by-ai",
    "https://www.bloomberg.com/news/articles/2026-02-03/stocks-rally-at-risk-as-retail-fervor-fades-says-citadel-securities-rubner",
    "https://www.bloomberg.com/news/articles/2026-02-03/-get-me-out-traders-dump-software-stocks-as-ai-fears-take-hold",
    "https://www.bloomberg.com/news/articles/2026-02-03/guard-against-ai-exuberance-in-credit-hsbc-strategists-say",
    "https://www.bloomberg.com/news/articles/2026-02-03/chip-designer-montage-is-said-set-to-price-902-million-hong-kong-listing-at-top",
    "https://www.bloomberg.com/news/articles/2026-02-03/samsung-sk-hynix-to-top-value-of-chinese-duo-as-ai-boom-shifts",
    "https://www.bloomberg.com/news/articles/2026-02-03/sitime-is-said-to-near-3-billion-deal-for-renesas-timing-unit",
    "https://www.bloomberg.com/news/articles/2026-02-03/overland-ai-raises-100m-to-speed-up-use-of-military-land-robots",
    "https://www.bloomberg.com/news/articles/2026-02-03/kuwait-opening-up-oil-fields-pipelines-to-foreign-investment",
    "https://www.bloomberg.com/news/articles/2026-02-03/russian-refiners-see-relief-from-decline-in-ukrainian-attacks",
    "https://www.bloomberg.com/news/articles/2026-02-03/elon-musk-s-spacex-said-to-combine-with-xai-ahead-of-mega-ipo",
    "https://www.bloomberg.com/news/articles/2026-02-03/france-eyes-shared-tools-at-g-7-to-address-economic-imbalances",
    "https://www.bloomberg.com/news/articles/2026-02-03/us-plans-to-issue-license-for-companies-to-pump-venezuelan-oil",
    "https://www.bloomberg.com/news/articles/2026-02-03/mexico-unveils-energy-focused-investment-plan-to-juice-economy",
    "https://www.bloomberg.com/news/articles/2026-02-02/latest-oil-market-news-and-analysis-for-february-3",
    "https://www.bloomberg.com/news/articles/2026-02-03/latest-oil-market-news-and-analysis-for-feb-3",
    "https://www.bloomberg.com/news/articles/2026-02-03/washington-fusion-startup-avalanche-energy-raises-29-million",

    # ── Reuters (Financial) ───────────────────────────────────────────────────
    "https://www.reuters.com/world/spain-hold-social-media-executives-accountable-illegal-hateful-content-2026-02-03/",
    "https://www.reuters.com/world/europe/palantir-ceo-defends-surveillance-tech-us-government-contracts-boost-sales-2026-02-02/",
    "https://www.reuters.com/legal/litigation/judge-fines-lawyers-12000-over-ai-generated-submissions-patent-case-2026-02-03/",

    # ── India Markets & Trade ─────────────────────────────────────────────────
    "https://economictimes.indiatimes.com/markets/ipos/fpos/indias-first-ai-ipo-fractal-analytics-announces-dates-for-rs-2834-crore-public-issue/articleshow/127889300.cms",
    "https://economictimes.indiatimes.com/markets/us-stocks/news/data-service-stocks-plunge-up-to-10-as-anthropic-releases-ai-in-legal-space/articleshow/127890436.cms",
    "https://economictimes.indiatimes.com/markets/stocks/news/why-is-stock-market-rising-today-rs-13-lakh-crore-boom-sensex-jumps-3500-pts-nifty-soars-nearly-5-india-us-trade-deal-among-factors-behind-rally/articleshow/127875336.cms",
    "https://economictimes.indiatimes.com/markets/forex/rupee-climbs-sharply-rallies-1-to-90-40-vs-dollar-in-early-trade/articleshow/127875859.cms",
    "https://economictimes.indiatimes.com/markets/commodities/news/gold-rebounds-more-than-3-after-sharp-selloff/articleshow/127874139.cms",
    "https://economictimes.indiatimes.com/news/economy/policy/rbi-to-start-3-day-deliberations-on-interest-rate-from-wednesday/articleshow/127887970.cms",
    "https://economictimes.indiatimes.com/industry/banking/finance/adani-green-to-seek-board-nod-to-raise-up-to-1-billion/articleshow/100459690.cms",
    "https://economictimes.indiatimes.com/wealth/tax/budget-2026-income-tax-highlights-changes-in-taxation-on-sgb-redemption-share-buyback-tcs-stt-nri-property-tds-updated-income-tax-slabs-standard-deduction-and-more/articleshow/127845285.cms",
    "https://www.moneycontrol.com/news/recommendations/reduce-aditya-birla-fashionretail-targetrs-230-emkay-global-financial_17531571.html",
    "https://www.moneycontrol.com/news/recommendations/buy-bajaj-finance-targetrs-9000-emkay-global-financial_17531631.html",
    "https://www.livemint.com/news/world/indiaus-trade-deal-trump-claims-india-will-stop-buying-russian-venezuelan-oil-what-we-know-so-far-11770106667644.html",
    "https://www.livemint.com/news/world/russias-first-comment-after-trumps-claim-on-india-us-trade-deal-11770115991320.html",
    "https://www.livemint.com/news/adani-us-sec-case-hires-trump-lawyer-robert-giuffra-11770091797614.html",
    "https://timesofindia.indiatimes.com/business/india-business/india-us-trade-deal-decoded-what-does-it-mean-for-economy-markets-russian-oil-imports-explained-in-10-charts/articleshow/127911573.cms",
    "https://timesofindia.indiatimes.com/business/india-business/why-trump-administration-is-still-imposing-18-tariff-on-india-explained/articleshow/127890865.cms",
    "https://timesofindia.indiatimes.com/business/india-business/why-union-budget-2026-avoided-populism-and-chose-stability/articleshow/127887080.cms",
    "https://timesofindia.indiatimes.com/business/india-business/jaishankar-rubio-talks-signal-strategic-reset-as-india-us-eye-critical-minerals-pact/articleshow/127914299.cms",
    "https://timesofindia.indiatimes.com/business/international-business/if-india-stops-buying-russian-oil-what-it-could-mean-for-russias-revenues-explained/articleshow/127914097.cms",
    "https://timesofindia.indiatimes.com/business/india-business/indian-it-stocks-crash-infosys-tcs-wipro-down-up-to-6-why-launch-of-new-ai-tool-by-us-startup-anthropic-is-driving-the-fall/articleshow/127899406.cms",
    "https://timesofindia.indiatimes.com/technology/tech-news/explained-what-is-anthropics-ai-tool-that-wiped-285-billion-off-software-stocks-in-a-single-day/articleshow/127892310.cms",

    # ── Tech / AI (market-moving, earnings impact) ────────────────────────────
    "https://techcrunch.com/2026/02/04/exclusive-positron-raises-230m-series-b-to-take-on-nvidias-ai-chips/",
    "https://techcrunch.com/2026/02/04/elevenlabs-raises-500m-from-sequioia-at-a-11-billion-valuation/",
    "https://techcrunch.com/2026/02/04/alexa-amazons-ai-assistant-is-now-available-to-everyone-in-the-u-s/",
    "https://techcrunch.com/2026/02/02/waymo-raises-16-billion-round-to-scale-robotaxi-fleet-london-tokyo/",
    "https://techcrunch.com/2026/02/02/what-snowflakes-deal-with-openai-tells-us-about-the-enterprise-ai-race/",
    "https://techcrunch.com/2026/02/02/elon-musk-spacex-acquires-xai-data-centers-space-merger/",
    "https://arstechnica.com/ai/2026/02/five-months-later-nvidias-100-billion-openai-investment-plan-has-fizzled-out/",
    "https://arstechnica.com/ai/2026/02/spacex-acquires-xai-plans-1-million-satellite-constellation-to-power-it/",
    "https://arstechnica.com/ai/2026/02/senior-staff-departing-openai-as-firm-prioritizes-chatgpt-development/",
    "https://www.ft.com/content/5038f2b1-6334-4d28-85e6-312d06796ca7",
    "https://www.technologyreview.com/2026/02/02/1131822/the-crucial-first-step-for-designing-a-successful-enterprise-ai-system/",
    "https://www.technologyreview.com/2026/02/04/1132115/the-download-the-future-of-nuclear-power-plants-and-social-media-fueled-ai-hype/",
    "https://www.forbes.com/sites/annatong/2026/02/02/the-top-open-ai-models-are-chinese-arcee-ai-thinks-thats-a-problem/",
]
