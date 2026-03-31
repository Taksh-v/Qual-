import json
import requests
import sys

def test_stream():
    url = "http://localhost:8000/intelligence/stream"
    payload = {
        "question": "What is the market impact of rising 10Y yields?",
        "geography": "US",
        "horizon": "MEDIUM_TERM",
        "response_mode": "detailed",
        "indicator_overrides": {}
    }
    
    print(f"Testing SSE stream at {url}...")
    try:
        with requests.post(url, json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"Error: HTTP {response.status_code}")
                print(response.text)
                sys.exit(1)
                
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("event: "):
                        print(f"\n[{decoded}]")
                    elif decoded.startswith("data: "):
                        data_str = decoded[6:]
                        try:
                            # If it's a token, print it nicely
                            evt_data = json.loads(data_str)
                            if "text" in evt_data:
                                print(evt_data["text"], end="", flush=True)
                            else:
                                print(f"  {data_str[:150]}...")
                        except Exception:
                            print(f"  {data_str[:150]}...")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Is FastAPI running on port 8000?")

if __name__ == "__main__":
    test_stream()
