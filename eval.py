import os
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_anthropic import ChatAnthropic
from rag import rag_chain

load_dotenv()

client = Client()
judge = ChatAnthropic(
    model="claude-haiku-4-5",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=100,
)

examples = [
    {
        "inputs": {"question": "What are the payment terms in the Accenture supply agreement?"},
        "outputs": {"answer": "Payment is due within 45 days of receipt of invoice. Invoices must be issued within 5 business days of month end."},
    },
    {
        "inputs": {"question": "Who are the parties in the ASML litigation?"},
        "outputs": {"answer": "ASML Holding NV as claimant and Precision Components GmbH as defendant."},
    },
    {
        "inputs": {"question": "What data protection obligations exist?"},
        "outputs": {"answer": "Prohibition on processing beyond documented purpose, sub-processor notification within 14 days, data breach notification within 24 hours."},
    },
    {
        "inputs": {"question": "What happens if an invoice is disputed?"},
        "outputs": {"answer": "Accenture must notify the supplier in writing within 15 days of a disputed invoice."},
    },
    {
        "inputs": {"question": "What are the termination rights?"},
        "outputs": {"answer": "Either party may terminate for material breach with written notice."},
    },
    {
        "inputs": {"question": "Which agreements require written notice for termination?"},
        "outputs": {"answer": "Both the Accenture NDA and Dutch supply agreement require written notice. The NDA requires 30 days notice. The Dutch supply agreement requires 60 working days notice tied to renewal periods."},
    },
    {
        "inputs": {"question": "What is the penalty for late delivery?"},
        "outputs": {"answer": "I don't have enough information to answer that."},
    },
]

dataset_name = "legal-rag-eval-v1"
existing = list(client.list_datasets(dataset_name=dataset_name))
if not existing:
    print("Creating dataset...")
    dataset = client.create_dataset(dataset_name)
    client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        dataset_id=dataset.id,
    )
    print("Dataset created.")
else:
    print("Dataset already exists, skipping creation.")

def predict(inputs):
    return {"answer": rag_chain.invoke(inputs["question"])}

def score_answer(run, example):
    predicted = run.outputs.get("answer", "")
    expected = example.outputs.get("answer", "")
    question = example.inputs.get("question", "")
    
    prompt = f"""You are evaluating a RAG pipeline answer.
Question: {question}
Expected answer: {expected}
Predicted answer: {predicted}

Is the predicted answer correct and complete compared to the expected answer?
Reply with only: CORRECT or INCORRECT"""

    response = judge.invoke(prompt)
    score = 1 if "CORRECT" in response.content.upper() else 0
    return {"key": "correctness", "score": score}

print("Running evaluation...")
results = evaluate(
    predict,
    data=dataset_name,
    evaluators=[score_answer],
    experiment_prefix="baseline",
)
print("Evaluation complete. Check LangSmith for results.")