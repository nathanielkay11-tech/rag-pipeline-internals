import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from eval import examples
import rag

load_dotenv()

THRESHOLD = 6
TOKEN_GATE_FACTOR = 1.5  # reject fix if avg input tokens increase by more than 50%


class InputTokenTracker(BaseCallbackHandler):
    def __init__(self):
        self.input_tokens = 0

    def on_llm_end(self, response, **kwargs):
        for gen_list in response.generations:
            for gen in gen_list:
                if hasattr(gen, "message") and gen.message.usage_metadata:
                    self.input_tokens += gen.message.usage_metadata.get(
                        "input_tokens", 0
                    )

judge = ChatAnthropic(
    model="claude-haiku-4-5",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=100,
)

_JUDGE_PROMPT = (
    "You are evaluating a RAG pipeline answer.\n"
    "Question: {question}\n"
    "Expected answer: {expected}\n"
    "Predicted answer: {predicted}\n\n"
    "Is the predicted answer correct and complete compared to the expected answer?\n"
    "Reply with only: CORRECT or INCORRECT"
)


def run_evals() -> dict:
    results = []
    for ex in examples:
        question = ex["inputs"]["question"]
        expected = ex["outputs"]["answer"]
        tracker = InputTokenTracker()
        predicted = rag.rag_chain.invoke(
            question, config={"callbacks": [tracker]}
        )
        prompt = _JUDGE_PROMPT.format(
            question=question, expected=expected, predicted=predicted
        )
        response = judge.invoke(prompt)
        score = 0 if "INCORRECT" in response.content.upper() else 1
        results.append(
            {
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "score": score,
                "input_tokens": tracker.input_tokens,
            }
        )

    total = sum(r["score"] for r in results)
    avg_input_tokens = sum(r["input_tokens"] for r in results) / len(results)
    return {
        "results": results,
        "score": total,
        "total": len(examples),
        "passed": total >= THRESHOLD,
        "failing": [r for r in results if r["score"] == 0],
        "avg_input_tokens": avg_input_tokens,
    }


if __name__ == "__main__":
    output = run_evals()
    print(f"\nScore: {output['score']}/{output['total']}  |  avg input tokens: {output['avg_input_tokens']:.0f}")
    for r in output["results"]:
        status = "PASS" if r["score"] == 1 else "FAIL"
        print(f"  [{status}] [{r['input_tokens']}t] {r['question'][:60]}")
