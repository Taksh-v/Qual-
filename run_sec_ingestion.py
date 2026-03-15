import os
import json
import logging
from datetime import datetime
from ingestion.sec_extractor import SECExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_DIR = "data/raw/sec"
os.makedirs(RAW_DIR, exist_ok=True)

def main():
    logger.info("Starting SEC EDGAR 8-K ingestion...")
    extractor = SECExtractor()
    filings = extractor.fetch_latest_8k()
    
    saved_count = 0
    for filing in filings:
        # Create a safe filename from the URL or title
        safe_title = "".join(c for c in filing['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_title[:50]}_{int(datetime.now().timestamp())}.json"
        
        filepath = os.path.join(RAW_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(filing, f, indent=2)
        saved_count += 1
        
    logger.info(f"Successfully saved {saved_count} new 8-K filings to {RAW_DIR}")

if __name__ == "__main__":
    main()
