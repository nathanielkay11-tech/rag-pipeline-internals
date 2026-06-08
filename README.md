# RAG Pipeline Internals

This is the fourth project in my AWS and AI portfolio. My previous three projects built progressively more intelligent cloud infrastructure on AWS — the last of which was a production RAG pipeline using Bedrock Knowledge Bases and OpenSearch Serverless. This project goes one layer deeper: stripping away the managed services to understand what's actually happening inside a RAG pipeline, and using LangSmith to instrument, observe, and evaluate every step.

The goal was not to build something production-grade. The goal was to understand the internals well enough to diagnose, debug, and improve RAG systems in any environment — which is the skill that sits underneath the architecture.

---

## 🗺️ What This Project Covers

| Phase | What I Built | What I Learned |
|---|---|---|
| Ingest | PDF loader → chunker → Voyage embeddings → Chroma | How documents become searchable vectors |
| Retrieval | Semantic similarity search against local vector store | Why retrieval fails before generation even starts |
| Generation | Retrieved context + question → Claude Haiku | How the prompt is actually constructed |
| Observability | LangSmith tracing on every step | How to see inside a pipeline and diagnose failures |
| Evaluation | Golden dataset + Claude-as-judge scoring | How to measure pipeline quality — and why eval scores lie |

---

## 🏗️ Stack

| Component | Tool | Why |
|---|---|---|
| Vector store | Chroma (local) | Runs entirely on-machine — no cloud, no cost, no data leaving the machine |
| Embeddings | Voyage AI `voyage-law-2` | Anthropic's recommended embedding provider; legal-domain model matched to the document set |
| Generation | Claude Haiku via Anthropic API | Cheapest Claude model; sufficient for Q&A against retrieved context |
| Orchestration | LangChain | Industry-standard orchestration layer; first-class LangSmith integration |
| Observability | LangSmith | Purpose-built tracing and evaluation for LLM pipelines |

---

## 💡 Why Local Instead of AWS

My previous project used Bedrock Knowledge Bases and OpenSearch Serverless — managed services that abstract the internals. That was the right call for a portfolio project: ship fast, demonstrate architectural thinking, prove cost efficiency.

This project inverts that. By running everything locally with no managed services, every step is visible:

- What does a vector actually look like?
- How does similarity scoring work, and why does it sometimes return the wrong chunks?
- What does the prompt look like when it hits Claude — not an abstraction of it, the actual text?
- Why does a vague question produce a worse answer than a specific one?

These are the questions that matter when a client's RAG pipeline is producing bad answers. Managed services hide the answers. This project surfaces them.

**The concepts transfer directly to any platform** — AWS, Azure, GCP, or a client's custom stack. The infrastructure changes. The retrieval mechanics don't.

---

## 🔒 Security and Cost

**Security**
- All API keys stored in `.env` — excluded from version control via `.gitignore`
- Documents stored locally — never committed to the repo, never leave the machine
- Chroma vector store is entirely local — no data transmitted except embedding and generation API calls

**Cost**
- Voyage AI: free tier (200M tokens/month) — this project used approximately 50K tokens total
- Claude Haiku: ~$0.001 per query — entire eval suite of 7 questions cost $0.0101
- No always-on infrastructure — everything runs as a manual Python script

---

## 📊 Eval Results — Baseline

**Dataset:** 7 questions across four categories — specific, vague, multi-document, and adversarial

**Score:** 7/7 (1.00) — but this is inflated. See Known Issues below.

**Average latency:** 2.69s per query

**Total cost for full eval run:** $0.0101

| Question Type | Example | Result |
|---|---|---|
| Specific | Payment terms in Accenture supply agreement | ✅ Correct |
| Specific | Parties in ASML litigation | ✅ Correct |
| Vague | What data protection obligations exist? | ✅ Correct |
| Specific | What happens if an invoice is disputed? | ✅ Correct |
| Vague | What are the termination rights? | ✅ Correct |
| Multi-document | Which agreements require written notice for termination? | ⚠️ Wrong answer, scored correct — see Known Issues |
| Adversarial | What is the penalty for late delivery? | ✅ Correctly returned "I don't have enough information" |

---

## ⚠️ Known Issues and Proposed Fixes

### Issue 1 — Judge prompt is too lenient
The LLM-as-judge evaluator scored question 6 (multi-document termination) as CORRECT when the answer was wrong. Claude's answer named different documents and different clauses than the expected answer. The judge saw "written notice" in both and called it a match.

**Root cause:** The judge prompt only checks for general correctness, not specific entity and number matching.

**Proposed fix:** Tighten the judge prompt with explicit scoring conditions:
- All specific document names must match
- All specific numbers and timeframes must match
- No information in the predicted answer may contradict the expected answer

**Lesson:** A 100% eval score doesn't mean the pipeline is working. It might mean the judge is broken. This is exactly why human spot checks matter even when automation is in place.

---

### Issue 2 — Fixed character chunking splits mid-clause
The current chunking strategy splits documents at 512 characters with 50-character overlap. Legal documents are structured by clause — a 512-character cut point frequently falls mid-sentence, producing chunks that are incomplete thoughts and therefore poorly retrievable.

**Evidence:** The disputed invoices answer in question 1 included the note "the context appears to be cut off" — Claude correctly flagged that it received half a clause.

**Proposed fix:** Switch from `RecursiveCharacterTextSplitter` with fixed character count to section-based chunking that splits on legal document structure markers (`Clause`, `Section`, `Article`). This produces self-contained, independently retrievable chunks.

---

### Issue 3 — No reranking
Chroma returns chunks ranked by vector similarity — meaning "sounds like the question" — not by whether the chunk actually answers the question. For vague questions, the wrong document type frequently ranks highest.

**Evidence:** The data protection question returned 3 chunks from a legal opinion (recommended amendments to a DPA) and 1 chunk from the Accenture supply agreement. The legal opinion ranked higher because it contained more data protection terminology — but it didn't contain the actual answer.

**Proposed fix:** Add a reranking step between retrieval and generation. After Chroma returns N candidates, a reranker model scores each for actual relevance to the question and reorders before Claude sees them. Voyage AI's `rerank-2` model pairs directly with the `voyage-law-2` embeddings used here.

**Why reranking matters at scale:** For a small corpus with specific questions, retrieval works well enough. For a large corpus with vague questions — which is realistic user behaviour — reranking is effectively required. Design for the worst question, not the best one.

---

### Issue 4 — No metadata tagging at ingest
Document chunks carry minimal metadata: source filename and page number. There is no document type tag (contract, legal opinion, litigation filing, correspondence).

**Impact:** Chroma cannot filter by document type during retrieval. A question about vendor obligations may return chunks from a legal opinion rather than a contract — because both contain similar legal language, but only one contains the actual obligations.

**Proposed fix:** Tag each document at ingest with `document_type`, `matter_id`, and `jurisdiction`. Retrieval queries can then filter by document type before similarity search, dramatically reducing noise for typed queries.

---

## 🔬 What LangSmith Shows You

LangSmith traces every step of every query. For each run you can see:

**VectorStoreRetriever** — The exact chunks returned by Chroma, their source documents, and their ranking order. This is where retrieval failures are visible. If the chunk that contains the answer is in position 3 of 4, reranking would have helped.

**ChatAnthropic** — The full prompt sent to Claude including all retrieved context, and Claude's raw response. This is where prompt construction failures are visible.

**Token counts and cost** — Per query, visible in the trace. Useful for understanding what's driving cost at scale.

**Eval experiments** — A scored table comparing predicted answers against expected answers, with per-question pass/fail and aggregate scores. The baseline for this project is shown above.

**The diagnostic workflow:**
1. Answer looks wrong → check VectorStoreRetriever → were the right chunks returned?
2. Right chunks returned but answer still wrong → check ChatAnthropic Input → was the prompt constructed correctly?
3. Prompt looks right but answer is wrong → model issue

This order covers the majority of RAG failures. Step 1 is where most problems actually live.

---

## 🤖 Development Approach

This project was built using an AI-assisted workflow. Claude (Anthropic) was used as a technical collaborator throughout — helping with code implementation, troubleshooting, and documentation. All architectural decisions, diagnostic reasoning, and learning objectives were directed by me.

This reflects how technical professionals actually work in 2026. Knowing how to leverage AI tools effectively — and critically evaluate their output — is itself a professional skill.

---

## 📁 Repository Structure

```
rag-pipeline-internals/
├── .env                 ← API keys — never committed
├── .gitignore
├── requirements.txt
├── ingest.py            ← PDF loader, chunker, Voyage embeddings, Chroma write
├── rag.py               ← Retriever, prompt, Claude Haiku, LangSmith tracing
├── eval.py              ← Golden dataset, Claude-as-judge scoring, LangSmith experiments
└── test-data/           ← Synthetic legal documents — never committed
```

---

## 🚀 How to Run

**Prerequisites:** Python 3.9+, API keys for Anthropic, Voyage AI, and LangSmith

```bash
git clone https://github.com/nathanielkay11-tech/rag-pipeline-internals.git
cd rag-pipeline-internals
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` with your keys:
```
ANTHROPIC_API_KEY=
VOYAGE_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=rag-pipeline-internals
LANGSMITH_ENDPOINT=https://aws.api.smith.langchain.com
```

```bash
python3 ingest.py   # Load documents into Chroma
python3 rag.py      # Run test queries — traces appear in LangSmith
python3 eval.py     # Run eval suite — results appear in LangSmith Datasets & Experiments
```
