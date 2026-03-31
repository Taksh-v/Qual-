import asyncio
from intelligence.agentic_rag.orchestrator import AgenticOrchestrator
from intelligence.bloomberg_formatter import BloombergFormatter
import time

async def main():
    print("Initializing AgenticOrchestrator...")
    orchestrator = AgenticOrchestrator()
    question = "What is the impact of rising 10-year yields on tech stocks?"
    
    print(f"Running query: {question}")
    
    t0 = time.time()
    final_event = None
    async for event in orchestrator.run_async(question):
        print(f"Event: {event.stage} (Agent: {event.agent_name})")
        if event.stage == "final":
            final_event = event
            
    print(f"Completed in {time.time() - t0:.2f}s")
    
    # We want to format the JSON answer into Bloomberg UI
    if final_event:
        answer_json = final_event.data["answer"]
        formatter = BloombergFormatter()
        
        # Mock some indicators and regime to see the full UI
        indicators = {"vix": 18.5, "yield_10y": 4.5, "sp500": 5100, "credit_hy": 350}
        regime = {"regime": "RISK_OFF"}
        cross_asset = {"overall_signal": "BEARISH"}
        
        output = formatter.morning_note(
            answer=answer_json,
            indicators=indicators,
            regime=regime,
            cross_asset=cross_asset,
            question=question,
            geography="US",
            horizon="MEDIUM_TERM"
        )
        print("\n" + "="*80)
        print("BLOOMBERG TERMINAL OUTPUT:")
        print("="*80 + "\n")
        print(output)
        print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(main())
