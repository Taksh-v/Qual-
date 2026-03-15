import os, sys, json, time
BASE = "/home/kali/Downloads/Qual"
os.chdir(BASE)
sys.path.insert(0, BASE)

# --- 1. Ollama ---
print("\n=== 1. Ollama / LLM ===")
try:
    import requests
    r = requests.get("http://localhost:11434/api/tags", timeout=4)
    models = [m["name"] for m in r.json().get("models", [])]
    print("OK  Ollama UP. Models:", models)
    ollama_ok = True
except Exception as e:
    print("ERR Ollama DOWN:", e); ollama_ok = False

if ollama_ok:
    try:
        import requests
        t0=time.time()
        er = requests.post("http://localhost:11434/api/embeddings",
            json={"model":"nomic-embed-text","prompt":"test"}, timeout=45)
        emb = er.json().get("embedding",[])
        print(f"OK  Embedding dim={len(emb)} latency={int((time.time()-t0)*1000)}ms")
    except Exception as e:
        print("WARN Embed failed:", e)

# --- 2. Vector store ---
print("\n=== 2. Vector Store ===")
INDEX_P = [os.path.join(BASE,"data","vector_db","news.index"),
           os.path.join(BASE,"index","faiss.index")]
META_P  = [os.path.join(BASE,"data","vector_db","metadata_with_entities.json"),
           os.path.join(BASE,"data","vector_db","metadata.json"),
           os.path.join(BASE,"index","metadata.json")]
idx_path = next((p for p in INDEX_P if os.path.exists(p)), None)
meta_path = next((p for p in META_P  if os.path.exists(p)), None)
if idx_path:
    import faiss
    idx = faiss.read_index(idx_path)
    print(f"OK  FAISS index: vectors={idx.ntotal}  dim={idx.d}")
else:
    print("ERR No FAISS index found"); idx=None
if meta_path:
    meta = json.load(open(meta_path, encoding="utf-8"))
    print(f"OK  Metadata: {len(meta)} chunks  file={os.path.basename(meta_path)}")
else:
    print("ERR No metadata file"); meta=[]

# --- 3. Retrieval ---
print("\n=== 3. Context Retrieval ===")
if idx and meta and ollama_ok:
    try:
        from intelligence.context_retriever import retrieve_relevant_context
        t0=time.time()
        chunks = retrieve_relevant_context("inflation outlook US", top_k=5, keep_latest=3)
        print(f"OK  {len(chunks)} chunks  {int((time.time()-t0)*1000)}ms")
        for i,c in enumerate(chunks[:2],1):
            print(f"   [{i}] {c.get('metadata',{}).get('title','N/A')[:60]}")
    except Exception as e:
        print("ERR Retrieval:", e)
else:
    print("SKIP (missing components)")

# --- 4. RAG pipeline ---
print("\n=== 4. RAG Pipeline (run_query) ===")
try:
    from rag.query import run_query
    import asyncio
    t0=time.time()
    answer, srcs = asyncio.run(run_query("What is the US inflation outlook?"))
    ms=int((time.time()-t0)*1000)
    print(f"OK  Completed in {ms}ms  sources={len(srcs)}")
    print("   Preview:", answer[:200].replace("\n"," "))
except Exception as e:
    import traceback; traceback.print_exc()
    print("ERR run_query:", e)

# --- 5. Macro pipeline ---
print("\n=== 5. Macro Intelligence Pipeline ===")
try:
    from intelligence.macro_engine import macro_intelligence_pipeline
    t0=time.time()
    out=""
    for chunk in macro_intelligence_pipeline("US equity outlook given inflation?", response_mode="brief"):
        out+=chunk
    ms=int((time.time()-t0)*1000)
    print(f"OK  Completed in {ms}ms  output={len(out)} chars")
    print("   Preview:", out[:400].replace("\n"," "))
except Exception as e:
    import traceback; traceback.print_exc()
    print("ERR macro_pipeline:", e)

print("\n=== Done ===\n")
