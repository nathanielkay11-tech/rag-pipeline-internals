import os
import subprocess
import tempfile
import py_compile
from datetime import datetime

RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
from dotenv import load_dotenv
import anthropic

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAG_PY = os.path.join(PROJECT_ROOT, "rag.py")
SONNET_MODEL = "claude-sonnet-4-6"


def _build_prompt(failing: list, rag_py: str) -> str:
    failing_block = "".join(
        f"\nQ: {item['question']}\n"
        f"Expected: {item['expected']}\n"
        f"Actual:   {item['predicted']}\n"
        for item in failing
    )
    return (
        "You are fixing a RAG pipeline that is failing evals. You may only edit rag.py.\n\n"
        "TUNABLE PARAMETERS — the only things you may change:\n"
        '- `k` in `vectorstore.as_retriever(search_kwargs={"k": <value>})` — controls chunk retrieval count\n'
        "- The prompt template string inside `ChatPromptTemplate.from_template(...)`\n\n"
        "READ-ONLY — do not modify under any circumstances:\n"
        "- eval.py\n"
        "- The golden dataset (the 7 question/answer pairs defined in eval.py)\n\n"
        "COST CONSTRAINT:\n"
        "- Prefer minimal k increases — do not raise k above 8\n"
        "- Prefer prompt template changes over k increases where possible\n\n"
        f"FAILING QUESTIONS ({len(failing)} of 7):\n"
        f"{failing_block}\n"
        "CURRENT rag.py:\n"
        f"{rag_py}\n\n"
        "Return ONLY the complete updated rag.py. No explanation, no markdown fences."
    )


def apply_fix(attempt: int, failing: list) -> str:
    with open(RAG_PY) as f:
        rag_py = f.read()

    prompt = _build_prompt(failing, rag_py)

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] Calling {SONNET_MODEL} — fix attempt {attempt}, {len(failing)} failing question(s)")
    response = client.messages.create(
        model=SONNET_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] Sonnet response received")
    updated_rag = response.content[0].text.strip()

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
            tmp.write(updated_rag)
            tmp_path = tmp.name
        py_compile.compile(tmp_path, doraise=True)
    except py_compile.PyCompileError as e:
        raise ValueError(f"Attempt {attempt}: Sonnet returned invalid Python — {e}") from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    branch = f"fix/loop-{RUN_ID}-attempt-{attempt}"
    subprocess.run(["git", "checkout", "-b", branch], check=True, cwd=PROJECT_ROOT)

    with open(RAG_PY, "w") as f:
        f.write(updated_rag)

    subprocess.run(["git", "add", "rag.py"], check=True, cwd=PROJECT_ROOT)
    subprocess.run(
        ["git", "commit", "-m", f"fix(loop): attempt {attempt} — auto-fix rag.py"],
        check=True,
        cwd=PROJECT_ROOT,
    )

    return branch
