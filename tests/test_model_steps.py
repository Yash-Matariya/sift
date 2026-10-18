"""
The two places the app hands text to the chat model outside the agent graph:
the answer chain's model step, and the query rewrite. Both must pass the prompt
through to generate() and return what it says.
"""
from app import agent, rag


def test_answer_chain_sends_the_rendered_prompt_to_the_model(monkeypatch):
    seen = {}

    def fake_generate(prompt):
        seen["prompt"] = prompt
        return "Refunds take 14 days."

    monkeypatch.setattr(rag, "generate", fake_generate)
    answer = rag.ANSWER_CHAIN.invoke({"context": "[1] Refunds take 14 days.", "question": "How long?"})
    assert answer == "Refunds take 14 days."
    assert "ONLY the context" in seen["prompt"]          # the guardrail reaches the model
    assert "[1] Refunds take 14 days." in seen["prompt"]
    assert "How long?" in seen["prompt"]


def test_rewrite_query_asks_the_model_and_strips_its_reply(monkeypatch):
    monkeypatch.setattr(agent, "generate", lambda prompt: "  refund policy timeline \n")
    assert agent.rewrite_query("how long do I wait for my money back?") == "refund policy timeline"


import pytest


@pytest.mark.parametrize("reply", [
    "I can't answer that. I'm happy to help with other questions though!",   # seen from llama3.1:8b
    "I’m sorry, but I cannot help with that.",
    "",
    "   \n  ",
    "Here is a query that should work well for this, because the question is about "
    "the refund policy and the time a refund takes to reach the customer's bank",
])
def test_a_reply_that_is_not_a_query_falls_back_to_the_question(monkeypatch, reply):
    monkeypatch.setattr(agent, "generate", lambda prompt: reply)
    assert agent.rewrite_query("What is the CEO's favourite colour?") == "What is the CEO's favourite colour?"


def test_quotes_and_trailing_explanation_are_trimmed(monkeypatch):
    monkeypatch.setattr(agent, "generate", lambda prompt: '"refund processing time"\nThis keeps the key terms.')
    assert agent.rewrite_query("How long does a refund take?") == "refund processing time"


def test_the_prompt_says_it_is_not_answering(monkeypatch):
    seen = {}
    monkeypatch.setattr(agent, "generate", lambda prompt: seen.setdefault("p", prompt) and "x")
    agent.rewrite_query("q")
    assert "NOT answering" in seen["p"]
