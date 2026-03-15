#!/usr/bin/env python3
"""
test_rss_live.py - Real-time RSS news fetching verification
"""
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

def test_rss_live():
    print("=" * 60)
    print("REAL-TIME NEWS FETCHING VERIFICATION TEST")
    print("=" * 60)

    # 1. Check dependencies
    print("\n[1] Checking dependencies...")
    deps = {}
    for pkg in ["feedparser", "bs4", "readability"]:
        try:
            __import__(pkg)
            deps[pkg] = "OK"
        except ImportError:
            deps[pkg] = "MISSING"
    for k, v in deps.items():
        status = "✓" if v == "OK" else "✗"
        print(f"  {status} {k}: {v}")

    # 2. Check feed configuration
    print("\n[2] Checking RSS feed configuration...")
    try:
        from config.rss_sources import RSS_FEEDS
        feeds = []
        for cat, items in RSS_FEEDS.items():
            for label, url in items:
                feeds.append((cat, label, url))
        print(f"  ✓ Total configured feeds: {len(feeds)}")
        cats = list(RSS_FEEDS.keys())
        print(f"  ✓ Categories: {cats}")
    except Exception as e:
        print(f"  ✗ Failed to load RSS config: {e}")
        return

    # 3. Live fetch test - fetch first 8 feeds
    print("\n[3] Live fetching test (first 8 feeds, no cache skip)...")
    try:
        from ingestion.rss_fetcher import fetch_all_feeds
        t0 = time.time()
        results = fetch_all_feeds(feeds[:8], max_workers=4, skip_seen=False)
        elapsed = time.time() - t0
        print(f"  ✓ Fetched {len(results)} articles in {elapsed:.1f}s")
        if results:
            print("\n  Sample articles:")
            for art in results[:5]:
                print(f"    [{art.get('category','?')}] {art.get('source','?')}")
                print(f"      Title: {art.get('title','?')[:80]}")
                print(f"      Date:  {art.get('date','?')}")
                text_len = len(art.get('raw_text',''))
                print(f"      Text length: {text_len} chars")
        else:
            print("  ✗ No articles fetched - possible network issue or all seen")
    except Exception as e:
        print(f"  ✗ Fetch failed: {e}")
        import traceback; traceback.print_exc()

    # 4. Verify Ollama connectivity
    print("\n[4] Checking Ollama LLM connectivity...")
    import requests as req_lib
    try:
        resp = req_lib.get("http://localhost:11434/api/tags", timeout=3)
        models = [m.get("name") for m in resp.json().get("models", [])]
        print(f"  ✓ Ollama reachable. Installed models: {models}")
    except Exception as e:
        print(f"  ✗ Ollama not reachable: {e}")

    # 5. Test model routing
    print("\n[5] Testing model router...")
    try:
        import sys
        sys.path.insert(0, "/home/kali/Downloads/Qual")
        from intelligence.model_router import get_model_candidates
        candidates = get_model_candidates()
        print(f"  ✓ Model candidates: {candidates}")
    except Exception as e:
        print(f"  ✗ Model router error: {e}")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_rss_live()
