import importlib
import subprocess
from datetime import datetime
import rag
from eval_runner import run_evals, THRESHOLD, TOKEN_GATE_FACTOR
from fix_agent import apply_fix

MAX_ATTEMPTS = 3


def _current_branch() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _checkout(branch: str) -> None:
    subprocess.run(["git", "checkout", branch], check=True)


def _reload_rag() -> None:
    importlib.reload(rag)


def _print_report(baseline: dict, attempts: list, success: bool) -> None:
    print("\n" + "=" * 60)
    print("LOOP REPORT")
    print("=" * 60)
    print(f"Baseline: {baseline['score']}/{baseline['total']} | avg input tokens: {baseline['avg_input_tokens']:.0f}")
    print()
    for a in attempts:
        score_str = f"{a['score']}/{baseline['total']}" if "score" in a else "n/a"
        token_str = f"{a['token_delta_pct']:+.1f}%" if "token_delta_pct" in a else "n/a"
        print(f"  Attempt {a['attempt']}: {a['outcome']:<22} | score: {score_str} | tokens: {token_str} | branch: {a.get('branch') or 'none'}")
    print()
    if success:
        winner = next(a for a in reversed(attempts) if a["outcome"] == "passed")
        print(f"SUCCESS — branch: {winner['branch']}")
        print(f"Score:  {baseline['score']}/{baseline['total']} → {winner['score']}/{baseline['total']}")
    else:
        print(f"FAILED — no fix found in {len(attempts)} attempt(s).")
    print("=" * 60)


def run_loop() -> None:
    base_branch = _current_branch()
    print(f"Base branch: {base_branch}\n")

    print("--- Baseline eval ---")
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] Calling run_evals() — baseline measurement")
    baseline = run_evals()
    print(f"Score: {baseline['score']}/{baseline['total']} | avg input tokens: {baseline['avg_input_tokens']:.0f}")

    if baseline["passed"]:
        print("Pipeline already passing threshold. Nothing to do.")
        return

    attempts = []
    current_failing = baseline["failing"]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n--- Fix attempt {attempt}/{MAX_ATTEMPTS} ---")

        try:
            branch = apply_fix(attempt, current_failing)
        except ValueError as e:
            print(f"Aborted: {e}")
            attempts.append({"attempt": attempt, "outcome": "invalid_python", "branch": None})
            _checkout(base_branch)
            _reload_rag()
            current_failing = baseline["failing"]
            continue

        _reload_rag()
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Calling run_evals() — scoring fix attempt {attempt}")
        result = run_evals()

        baseline_tokens = baseline["avg_input_tokens"] or 1
        token_ratio = result["avg_input_tokens"] / baseline_tokens
        token_delta_pct = (token_ratio - 1) * 100
        print(f"Score: {result['score']}/{result['total']} | avg input tokens: {result['avg_input_tokens']:.0f} ({token_delta_pct:+.1f}%)")

        if token_ratio > TOKEN_GATE_FACTOR:
            print(f"Token gate triggered ({token_delta_pct:+.1f}% > +50%) — rejecting.")
            attempts.append({"attempt": attempt, "outcome": "token_gate_rejected", "branch": branch,
                             "score": result["score"], "token_delta_pct": token_delta_pct})
            _checkout(base_branch)
            _reload_rag()
            current_failing = baseline["failing"]
            continue

        if result["passed"]:
            attempts.append({"attempt": attempt, "outcome": "passed", "branch": branch,
                             "score": result["score"], "token_delta_pct": token_delta_pct})
            _print_report(baseline, attempts, success=True)
            return

        attempts.append({"attempt": attempt, "outcome": "score_failed", "branch": branch,
                         "score": result["score"], "token_delta_pct": token_delta_pct})
        current_failing = result["failing"]
        _checkout(base_branch)
        _reload_rag()

    _print_report(baseline, attempts, success=False)


if __name__ == "__main__":
    run_loop()
