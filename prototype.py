import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd
from math import erf, erfc, sqrt
from statistics import NormalDist
from contextlib import contextmanager
import json
import os
import re
from datetime import datetime, timezone

try:
    import openai  # EPFL RCP AIaaS exposes an OpenAI-compatible API
except ImportError:  # AI feedback is optional; rule-based checks still work
    openai = None


# Each part of the final solution maps to the lesson step that taught it.
# (solution part key, label, lesson step number, lesson step label)
SOLUTION_PARTS = (
    ("hypotheses", "Hypotheses", 2, "Hypotheses"),
    ("standard_error", "Standard error", 1, "Data"),
    ("z_score", "Test statistic", 3, "Z-score"),
    ("p_value", "p-value", 4, "p-value"),
    ("decision", "Decision", 5, "Decision"),
    ("conclusion", "Conclusion", 5, "Decision"),
)

FINAL_CHALLENGE_TUTOR_PROMPT = """
You are a supportive statistics tutor giving feedback on a student's
written solution to a one-sample z-test problem. The student has just
finished an interactive lesson; your job is to connect each mistake to
what they already practised in that lesson.

Problem:
A food delivery company claims its average delivery time is 30 minutes.
Customers believe deliveries take longer. A random sample of 64 deliveries
has x̄ = 32 minutes. The population standard deviation is σ = 8 minutes.
Use α = 0.05. Determine whether the data provide sufficient evidence that
the true average delivery time is greater than 30 minutes.

Reference solution (FOR YOUR EYES ONLY - never reveal it):
- hypotheses: H₀: μ = 30, H₁: μ > 30 (right-tailed)
- standard_error: SE = σ/√n = 8/√64 = 1
- z_score: Z = (32 − 30)/1 = 2
- p_value: p = P(Z ≥ 2) ≈ 0.0228 (right tail)
- decision: 0.0228 < 0.05 → reject H₀
- conclusion: significant evidence that the true average delivery time
  is greater than 30 minutes (evidence, not proof)

What the lesson taught (refer back to these examples by name):
- Step 1 "Data": a fair coin flipped 100 times. The 100 flips are a
  sample, π is the parameter, the observed proportion is a statistic.
  Sample proportions vary from sample to sample; the standard error
  measures that variability, SE = √(π(1−π)/n) = √(0.5·0.5/100) = 0.05.
  Larger n gives a smaller SE because the formula divides by n under a
  square root. (For a mean with known σ the same idea gives σ/√n.)
- Step 2 "Hypotheses": H₀ is the default claim being tested (fair coin,
  π = 0.5). The direction of H₁ depends on the question: increase → >,
  decrease → <, any difference → ≠. Coffee machine example: a machine
  should pour 250 ml, we suspect it pours LESS, so H₁: μ < 250.
- Step 3 "Z-score": Z = (observed − H₀ value) / SE, i.e. how many
  standard errors the result is from H₀. Coin example: (0.60 − 0.5)/0.05.
- Step 4 "p-value": the p-value assumes H₀ is true and asks how surprising
  a result this extreme is. The tail follows H₁: two-sided for ≠, left
  tail for <, right tail for >. Coffee example: x̄ = 244, SE = 2, Z = −3,
  H₁: μ < 250 so p = P(Z ≤ −3) ≈ 0.00135. Farther from H₀ → smaller p.
- Step 5 "Decision": compare p with α. Coin example 0.046 < 0.05 → reject
  H₀; 0.12 > 0.05 → fail to reject H₀. We never "accept H₀" or "accept
  H₁", and significant evidence is not proof or certainty.

Rules:
- Grade each of the six parts: "correct", "incorrect", "missing", or
  "minor" (small wording/notation issue). "correct" only if the whole part
  is right. Accept equivalent notation and methods (e.g. "mu", "mean",
  critical value 2 > 1.645 instead of a p-value, rounding like 0.023).
  If an earlier mistake carries forward, say so, and judge the later part
  on whether the method was right.
- NEVER reveal the correct answer for a part that is incorrect or missing:
  no correct hypothesis, no correct symbol (>, <, ≠), no correct formula
  for this problem, no correct number, and no "it should be ...". Do not
  write a worked solution. The student must work it out by going back to
  the lesson. Instead, in 1-3 sentences:
  1. say what is wrong with what THEY wrote (quote their value),
  2. point to the matching lesson example and what it showed,
  3. end with a short guiding question.
  It is fine to quote formulas and numbers from the lesson examples
  (the coin and coffee machine), just not this problem's answers.
  Bad (reveals the answer): "H₁ should be μ > 30." / "Use SE = σ/√n."
  Good: "You used ≠, which tests for any difference. Remember the coffee
  machine in Step 2: we suspected it pours less, and that suspicion set
  the direction of H₁. What do the customers suspect here?"
  Good: "You divided σ by n. In Step 1, the coin's SE put n under a
  square root. How does that change your calculation?"
- For a correct part, give one short sentence of confirmation without
  restating the full calculation.
- Be encouraging. Write math inline with single dollar signs, e.g. $H_0$.

Reply with ONLY a JSON object, no other text, in exactly this shape:
{
  "parts": {
    "hypotheses": {"status": "...", "feedback": "..."},
    "standard_error": {"status": "...", "feedback": "..."},
    "z_score": {"status": "...", "feedback": "..."},
    "p_value": {"status": "...", "feedback": "..."},
    "decision": {"status": "...", "feedback": "..."},
    "conclusion": {"status": "...", "feedback": "..."}
  },
  "overall": "one encouraging sentence"
}
"""


def get_setting(name):
    """Read a setting from Streamlit secrets, falling back to env vars."""
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets.toml present
        value = None
    return value or os.environ.get(name)


def get_ai_config():
    """Return (api_key, base_url, model) or None if AI is not configured."""
    config = (
        get_setting("RCP_API_KEY"),
        get_setting("RCP_BASE_URL") or "https://inference.rcp.epfl.ch/v1",
        get_setting("RCP_MODEL"),
    )
    return config if all(config) else None


def clean_model_text(text):
    """Strip inline reasoning and convert LaTeX delimiters for Streamlit."""
    # Reasoning models (Qwen, gpt-oss) may include their thinking inline.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Streamlit renders $...$ / $$...$$ math but not \( \) / \[ \].
    text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    return text.strip()


# LaTeX commands whose first letter is also a JSON escape (\b \f \n \r \t).
LATEX_COMMANDS = (
    r"(?:frac|bar|beta|bigl|bigr|boxed|neq|ne|nu|not|rightarrow|right|rho|"
    r"tau|theta|times|text|tilde|top)(?![a-zA-Z])"
)


def fix_json_backslashes(raw):
    r"""Escape LaTeX backslashes (\mu, \frac, \uparrow) that break JSON.

    Valid JSON escapes are kept, except when \b \f \n \r \t start a known
    LaTeX command such as \frac, \neq or \theta.
    """
    def repair(match):
        escape = match.group(1)
        rest = raw[match.start() + 1:]
        if escape in ('\\', '"', '/'):
            return match.group(0)
        if len(escape) == 5:  # \uXXXX
            return match.group(0)
        if escape in "bfnrt" and not re.match(LATEX_COMMANDS, rest):
            return match.group(0)
        return "\\\\" + escape

    return re.sub(r"\\(u[0-9a-fA-F]{4}(?![a-zA-Z])|.)", repair, raw,
                  flags=re.DOTALL)


def get_ai_feedback(student_work, api_key, base_url, model):
    """Ask the EPFL RCP model to grade the final challenge.

    Returns a dict: {"parts": {key: {"status", "feedback"}}, "overall": str}.
    """
    data = call_llm_json(
        FINAL_CHALLENGE_TUTOR_PROMPT,
        [{
            "role": "user",
            "content": (
                "Here is the student's solution:\n\n"
                f"<student_work>\n{student_work}\n</student_work>"
            ),
        }],
        (api_key, base_url, model),
    )

    parts = {}
    for key, *_ in SOLUTION_PARTS:
        part = data.get("parts", {}).get(key) or {}
        status = str(part.get("status", "missing")).lower()
        if status not in ("correct", "incorrect", "missing", "minor"):
            status = "incorrect"
        parts[key] = {
            "status": status,
            "feedback": clean_model_text(str(part.get("feedback", ""))),
        }
    return {
        "parts": parts,
        "overall": clean_model_text(str(data.get("overall", ""))),
    }


# ============================================================
# Study setup (M4): condition, test-form order and participant ID
# ============================================================

CONDITIONS = ("ai", "control")
FORM_ORDERS = ("AB", "BA")

# The Step 8 AI feedback is switched off: Step 8 must be identical in both
# versions (rule-based check), so that only F1-F3 differ between them.
FINAL_CHALLENGE_AI_FEEDBACK = False


def is_dev_mode():
    """DEV_MODE = true in .streamlit/secrets.toml (never on the study app)."""
    return str(get_setting("DEV_MODE") or "").lower() in ("1", "true", "yes")


def initialize_study_settings():
    """Read ?c=, ?f= and ?id= from the link once per session.

    Returns False when the link is incomplete (and we are not in dev mode).
    """
    if "study" in st.session_state:
        return True

    condition = str(st.query_params.get("c", "")).strip().lower()
    form_order = str(st.query_params.get("f", "")).strip().upper()
    participant_id = str(st.query_params.get("id", "")).strip()
    id_ok = re.fullmatch(r"[A-Za-z0-9_-]{1,20}", participant_id) is not None

    if condition in CONDITIONS and form_order in FORM_ORDERS and id_ok:
        st.session_state.study = {
            "condition": condition,
            "form_order": form_order,
            "participant_id": participant_id,
        }
        return True

    if is_dev_mode():
        st.session_state.study = {
            "condition": condition if condition in CONDITIONS else "ai",
            "form_order": form_order if form_order in FORM_ORDERS else "AB",
            "participant_id": participant_id if id_ok else "DEV",
        }
        return True

    return False


def restart_activity():
    """Restart the lesson only.

    Keeps the participant's condition, forms, ID, consent and test answers,
    and returns to the lesson introduction (never to consent or pre-test).
    """
    kept = {
        key: st.session_state[key]
        for key in ("study", "study_data")
        if key in st.session_state
    }
    st.session_state.clear()
    st.session_state.update(kept)
    st.session_state.step = lesson_start_index()


# ============================================================
# Shared LLM call (EPFL RCP, OpenAI-compatible)
# ============================================================

def call_llm_json(system_prompt, messages, ai_config, attempts=2):
    """Send a chat to the model and parse its JSON reply into a dict.

    Retries once if the reply cannot be parsed.
    """
    api_key, base_url, model = ai_config
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    last_error = None
    for _ in range(attempts):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        text = response.choices[0].message.content or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        try:
            if not match:
                raise ValueError(
                    "The AI reply was not in the expected format.")
            return json.loads(fix_json_backslashes(match.group(0)))
        except (ValueError, json.JSONDecodeError) as error:
            last_error = error
    raise ValueError(f"The AI reply could not be read: {last_error}")


def ai_error_message(error):
    """Turn an exception from the AI call into a message for the page."""
    if isinstance(error, openai.AuthenticationError):
        return "The AI key is invalid. Check RCP_API_KEY."
    if isinstance(error, openai.NotFoundError):
        return "Model or endpoint not found. Check RCP_MODEL and RCP_BASE_URL."
    if isinstance(error, openai.RateLimitError):
        return "The AI tutor is busy right now. Wait a few seconds and try again."
    if isinstance(error, openai.APIConnectionError):
        return "Could not reach the AI tutor. Check the internet connection."
    if isinstance(error, (ValueError, json.JSONDecodeError)):
        return "The AI tutor's reply could not be read. Please try again."
    return f"The AI tutor failed: {error}"


# ============================================================
# Free-text prompts F1-F3: write → feedback → revise → model answer
# ============================================================
# Only move 2 (feedback) differs between the two versions:
#   ai      → LLM chat about this answer (max MAX_LEARNER_REPLIES replies)
#   control → one static explanation, the same for every learner
#
# DRAFT CONTENT: control explanations and model answers are placeholders
# for the team to finalise. Edit them here only.

MAX_LEARNER_REPLIES = 3

MISCONCEPTIONS = {
    "P_H0": (
        "p is the probability that H₀ is true",
        "There is a 4.6% chance the coin is fair.",
    ),
    "CHANCE": (
        "p is the probability that the result is a fluke / due to chance",
        "4.6% chance we got 60 heads by luck.",
    ),
    "ONE_MINUS_P": (
        "1 − p is the probability that H₁ is true",
        "95.4% sure the coin is biased.",
    ),
    "NO_CONDITION": (
        "right idea, but forgets that p is computed assuming H₀ is true",
        "The probability of getting 60 heads or more.",
    ),
    "NO_EXTREME": (
        "right idea, but forgets 'or more extreme' (talks about exactly "
        "this result)",
        "If the coin is fair, the chance of exactly 60 heads.",
    ),
    "HARD_THRESHOLD": (
        "treats 0.05 as a hard line, so p = 0.046 and p = 0.072 are seen "
        "as completely different findings",
        "Yes: 60 heads proves bias, 59 doesn't.",
    ),
    "PROVES_H0": (
        "thinks not rejecting H₀ proves H₀ is true",
        "With 59 heads the coin is fair.",
    ),
}

FREE_TEXT_PROMPTS = {
    "F1": {
        "question": (
            "We got **60 heads** in 100 flips, with **p ≈ 0.046**. "
            "In your own words, what does this p-value tell us?"
        ),
        "context": (
            "A coin is flipped 100 times and lands heads 60 times. "
            "H₀: the coin is fair (probability of heads 0.5). "
            "H₁: the coin is not fair (two-sided). Z = 2, two-sided "
            "p ≈ 0.046."
        ),
        "key_idea": (
            "If the coin were fair (H₀ true), a result at least as extreme "
            "as 60 heads (60 or more heads, or 40 or fewer) would happen "
            "about 4.6% of the time."
        ),
        "labels": ("P_H0", "CHANCE", "ONE_MINUS_P", "NO_CONDITION",
                   "NO_EXTREME"),
        "control_explanation": (
            "**How to read this p-value.** A p-value is always computed "
            "*assuming H₀ is true*. Here: if the coin were fair, results at "
            "least as extreme as 60 heads (60 or more heads, or 40 or fewer) "
            "would happen about 4.6% of the time.\n\n"
            "Common mistakes:\n"
            "- It is **not** the probability that the coin is fair "
            "(that H₀ is true).\n"
            "- It is **not** the probability that the result is a fluke or "
            "due to chance.\n"
            "- 1 − p = 95.4% is **not** the probability that the coin is "
            "biased (that H₁ is true).\n"
            "- It is about results **at least as extreme**, not exactly "
            "60 heads, and it only makes sense **if H₀ is true**."
        ),
        "model_answer": (
            "If the coin were fair, we would see a result at least as "
            "extreme as 60 heads (60 or more heads, or 40 or fewer) in about "
            "4.6% of experiments of 100 flips. Such a result is fairly "
            "unusual for a fair coin, which counts as evidence against H₀. "
            "It does not tell us the probability that the coin is fair."
        ),
    },
    "F2": {
        "question": (
            "What does **p ≈ 0.00135** mean for the coffee machine?"
        ),
        "context": (
            "A coffee machine should pour 250 ml on average; we suspect it "
            "pours less. Sample mean x̄ = 244 ml, SE = 2 ml, so Z = −3. "
            "H₀: μ = 250, H₁: μ < 250 (left-tailed). p = P(Z ≤ −3) ≈ 0.00135."
        ),
        "key_idea": (
            "If the machine really poured 250 ml on average (H₀ true), a "
            "sample mean of 244 ml or lower would happen only about 0.135% "
            "of the time."
        ),
        "labels": ("P_H0", "CHANCE", "ONE_MINUS_P", "NO_CONDITION",
                   "NO_EXTREME"),
        "control_explanation": (
            "**How to read this p-value.** It is computed *assuming H₀ is "
            "true*: if the machine really poured 250 ml on average, a sample "
            "mean of 244 ml **or lower** would happen only about 0.135% of "
            "the time.\n\n"
            "Common mistakes:\n"
            "- It is **not** the probability that the machine works "
            "correctly (that H₀ is true).\n"
            "- It is **not** the probability that the low mean is just luck "
            "or due to chance.\n"
            "- 1 − p = 99.865% is **not** the probability that the machine "
            "pours less (that H₁ is true).\n"
            "- It is about results **this low or lower**, not exactly "
            "244 ml, and it only makes sense **if H₀ is true**."
        ),
        "model_answer": (
            "If the machine really poured 250 ml on average, a sample mean "
            "of 244 ml or lower would happen only about 0.135% of the time. "
            "That would be very surprising, so the data are strong evidence "
            "that the machine pours less than 250 ml. It is not the "
            "probability that the machine works correctly."
        ),
    },
    "F3": {
        "question": (
            "**60 heads** gives **p ≈ 0.046**, **59 heads** gives "
            "**p ≈ 0.072**. Are these really different findings? Why?"
        ),
        "context": (
            "Coin flipped 100 times, H₀: fair coin, two-sided test, "
            "α = 0.05. 60 heads → p ≈ 0.046 (reject H₀); "
            "59 heads → p ≈ 0.072 (fail to reject H₀)."
        ),
        "key_idea": (
            "The two results are almost the same evidence: one more head "
            "changes p only a little. α = 0.05 is a convention, not a cliff. "
            "60 heads does not prove the coin is biased, and failing to "
            "reject with 59 heads does not prove the coin is fair."
        ),
        "labels": ("HARD_THRESHOLD", "PROVES_H0"),
        "control_explanation": (
            "**Comparing p ≈ 0.046 and p ≈ 0.072.** One extra head changes "
            "the p-value only a little, so the two results carry very "
            "similar evidence against a fair coin. α = 0.05 is a convention "
            "for making a decision, not a line where evidence suddenly "
            "appears.\n\n"
            "Common mistakes:\n"
            "- Treating 0.05 as a hard line: 60 heads does **not** prove "
            "the coin is biased, and 59 heads is not \"no evidence\".\n"
            "- Thinking that not rejecting H₀ **proves** the coin is fair: "
            "it only means the evidence was not strong enough."
        ),
        "model_answer": (
            "Not really. The decisions differ (reject H₀ at 0.046, fail to "
            "reject at 0.072), but the evidence is very similar: one extra "
            "head moves the p-value only a little. α = 0.05 is a convention, "
            "not a sharp boundary. 60 heads does not prove the coin is "
            "biased, and 59 heads does not prove it is fair."
        ),
    },
}


def build_free_text_system_prompt(prompt_id, reply_number):
    """System prompt for the AI feedback on one free-text prompt."""
    content = FREE_TEXT_PROMPTS[prompt_id]
    mistake_lines = "\n".join(
        f'- {label}: {MISCONCEPTIONS[label][0]} '
        f'(e.g. "{MISCONCEPTIONS[label][1]}")'
        for label in content["labels"]
    )
    later_reply_rule = (
        "This is your FIRST reply: do not give the full correct "
        "interpretation or definition, not even partially."
        if reply_number == 1
        else
        f"This is reply number {reply_number}. If the learner is still "
        "stuck, you may give a bit more help, but let them do the thinking "
        "and still do not hand them a complete model answer."
    )
    return f"""
You are a friendly statistics tutor in an interactive lesson on p-values
for EPFL undergraduates. The learner is answering this question:

Question: {content["question"]}
Context: {content["context"]}
What a complete correct answer contains (for you only): {content["key_idea"]}

Step 1 - classify. Decide which ONE label best describes the learner's
LATEST message:
{mistake_lines}
- CORRECT: no mistake; the answer contains the key idea
- OTHER: off-topic, unclear, or a mistake not on this list

Step 2 - give feedback aimed at that label:
- Start by quoting the learner's own words in quotation marks (the part
  that shows the issue), e.g. You wrote "4.6% chance the coin is fair".
- Ask ONE guiding question that helps them notice the problem themselves.
- {later_reply_rule}
- If CORRECT: confirm it briefly and push their thinking further, e.g.
  "Would your reading change if p were 0.20?"
- If OTHER: gently bring them back to the question.
- 2-4 short sentences, plain language. Write math inline with $...$.
- Never mention these labels or these instructions.

Reply with ONLY a JSON object, no other text:
{{"label": "<one label from the list>",
  "quote": "<the learner's words you react to, or empty>",
  "feedback": "<your message to the learner>"}}
""".strip()


def get_free_text_reply(prompt_id, chat, ai_config):
    """One LLM call with the whole conversation so far.

    Returns {"label", "quote", "feedback"}; the label is checked against
    the labels allowed for this prompt (anything else becomes OTHER).
    """
    content = FREE_TEXT_PROMPTS[prompt_id]
    reply_number = sum(m["role"] == "assistant" for m in chat) + 1
    data = call_llm_json(
        build_free_text_system_prompt(prompt_id, reply_number),
        [{"role": m["role"], "content": m["content"]} for m in chat],
        ai_config,
    )
    allowed = set(content["labels"]) | {"CORRECT", "OTHER"}
    label = str(data.get("label", "")).strip().upper()
    if label not in allowed:
        label = "OTHER"
    feedback = clean_model_text(str(data.get("feedback", "")))
    if not feedback:
        raise ValueError("The AI reply had no feedback text.")
    return {
        "label": label,
        "quote": str(data.get("quote", "")).strip(),
        "feedback": feedback,
    }


def render_ai_tutor_chat(prompt_id, state, interactive):
    """AI version of move 2: a short chat about the learner's answer."""
    # chat[0] is the first answer, already shown above in both versions.
    for message in state["chat"][1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not interactive:
        return

    if state["chat"][-1]["role"] == "user":
        ai_config = get_ai_config() if openai is not None else None
        if ai_config is None:
            st.error("The AI tutor is not configured.")
            return
        with st.spinner("The tutor is reading your answer..."):
            try:
                reply = get_free_text_reply(prompt_id, state["chat"],
                                            ai_config)
            except Exception as error:  # shown to the learner, retryable
                st.error(ai_error_message(error))
                st.button("Try again", key=f"ft_{prompt_id}_retry")
                return
        # label and quote are kept for the logs; the learner never sees them.
        state["chat"].append({
            "role": "assistant",
            "content": reply["feedback"],
            "label": reply["label"],
            "quote": reply["quote"],
        })
        st.rerun()

    learner_replies = sum(m["role"] == "user" for m in state["chat"]) - 1
    if learner_replies < MAX_LEARNER_REPLIES:
        with st.container():
            message = st.chat_input(
                f"Reply to the tutor ({MAX_LEARNER_REPLIES - learner_replies}"
                " left), or continue below",
                key=f"ft_{prompt_id}_chat_input",
            )
        if message and message.strip():
            state["chat"].append({"role": "user",
                                  "content": message.strip()})
            st.rerun()
    else:
        st.caption("You have used all your replies. Continue to revise "
                   "your answer below.")


def render_free_text_activity(prompt_id):
    """Four moves: write → feedback → revise → model answer.

    Only the feedback move depends on the condition. Each move is locked
    once submitted, also when the learner navigates away and back.
    Returns True when all four moves are done.
    """
    content = FREE_TEXT_PROMPTS[prompt_id]
    condition = st.session_state.study["condition"]
    state = st.session_state.setdefault("free_text", {}).setdefault(
        prompt_id,
        {"stage": "write", "first": "", "chat": [], "revised": ""},
    )

    st.markdown(content["question"])

    # Move 1: write a first answer (cannot be empty).
    if state["stage"] == "write":
        first = st.text_area("Your answer", key=f"ft_{prompt_id}_first",
                             height=120)
        if st.button("Submit answer", key=f"ft_{prompt_id}_submit_first",
                     type="primary"):
            if not first.strip():
                st.warning("Please write an answer first.")
            else:
                state["first"] = first.strip()
                state["chat"] = [{"role": "user", "content": state["first"]}]
                state["stage"] = "feedback"
                st.rerun()
        return False

    with st.container(border=True):
        st.caption("Your first answer")
        st.markdown(state["first"])

    # Move 2: feedback. THE ONLY PART THAT DIFFERS BETWEEN THE VERSIONS.
    st.markdown("**Feedback**")
    if condition == "ai":
        render_ai_tutor_chat(prompt_id, state,
                             interactive=state["stage"] == "feedback")
        feedback_ready = any(m["role"] == "assistant" for m in state["chat"])
    else:
        st.info(content["control_explanation"])
        feedback_ready = True

    if state["stage"] == "feedback":
        if st.button("Continue: revise my answer",
                     key=f"ft_{prompt_id}_to_revise",
                     type="primary", disabled=not feedback_ready):
            st.session_state[f"ft_{prompt_id}_revised"] = state["first"]
            state["stage"] = "revise"
            st.rerun()
        return False

    # Move 3: one required revised answer, pre-filled with the first one.
    if state["stage"] == "revise":
        revise_key = f"ft_{prompt_id}_revised"
        if revise_key not in st.session_state:
            st.session_state[revise_key] = state["first"]
        revised = st.text_area("Revise your answer", key=revise_key,
                               height=120)
        if st.button("Submit revised answer",
                     key=f"ft_{prompt_id}_submit_revised", type="primary"):
            if not revised.strip():
                st.warning("Please write your revised answer.")
            else:
                state["revised"] = revised.strip()
                state["stage"] = "done"
                st.rerun()
        return False

    with st.container(border=True):
        st.caption("Your revised answer")
        st.markdown(state["revised"])

    # Move 4: the same model answer in both versions.
    st.success(f"**Model answer.** {content['model_answer']}")
    return True


# ============================================================
# Study screens (M4): consent, pre-test, post-test, end
# ============================================================
# DRAFT CONTENT: info-sheet/consent text and all test wording below are
# drafts for the team to finalise. Edit them here only.

INFO_SHEET = """
**What this study is about.** You will work through a short interactive
lesson on hypothesis testing and p-values (about 30 minutes), with a short
test before and after it. We compare two versions of the lesson.

**What we record.** Your answers in the lesson and the tests, the time you
spend on each part and, where the lesson includes it, your messages with
an AI tutor. Data are linked only to a participant code, not to your name.

**Your rights.** Taking part is voluntary. You can stop at any time
without giving a reason, and ask us to delete your data. The data are used
only for the CS-411 Digital Education course project.

**Contact:**  Mattéo Le Cué - matteo.lecue@epfl.ch, Omar Idrissi - omar.idrissikaitouni@epfl.ch, Rana ElGahawy - rana.elgahawy@epfl.ch
"""

CONSENT_LABEL = (
    "I have read the information above and agree to take part."
)

SCREENING_QUESTION = (
    "Have you learned about hypothesis tests or p-values before?"
)

TEST_RULES = (
    "Answer on your own, without notes or a calculator. There is no "
    "feedback, and you cannot go back once you submit."
)

TRUE_FALSE = ("True", "False")

TEST_FORMS = {
    "A": {
        "scenario": (
            "A company tests a new **button** on its website. Visitors who "
            "see the new button click it more often than visitors who see "
            "the old one. The test gives **p = 0.03**."
        ),
        "q1": "What does this p-value mean?",
        "q2": (
            ("The probability that the new button has no effect (H₀ is "
             "true) is 3%.", "False"),
            ("The probability that the difference in clicks is due to "
             "chance is 3%.", "False"),
            ("If the new button had no effect (H₀ true), a difference in "
             "clicks at least this large would occur about 3% of the time.",
             "True"),
            ("There is a 97% probability that the new button really gets "
             "more clicks (H₁ is true).", "False"),
        ),
        "q3": (
            "The company repeats the test twice. One replication gives "
            "**p = 0.038**, the other gives **p = 0.062**. How different is "
            "the evidence from these two replications?"
        ),
        "q4": (
            "A colleague runs another test and gets **p = 0.08**. They "
            "conclude: *\"p = 0.08, so the new button has no effect.\"* "
            "Evaluate this claim."
        ),
    },
    "B": {
        "scenario": (
            "Students using a **new study method** score higher on an exam "
            "than students who do not use it. The test gives **p = 0.02**."
        ),
        "q1": "What does this p-value mean?",
        "q2": (
            ("The probability that the study method has no effect (H₀ is "
             "true) is 2%.", "False"),
            ("The probability that the difference in scores is due to "
             "chance is 2%.", "False"),
            ("If the study method had no effect (H₀ true), a difference in "
             "scores at least this large would occur about 2% of the time.",
             "True"),
            ("There is a 98% probability that the study method really "
             "improves scores (H₁ is true).", "False"),
        ),
        "q3": (
            "The study is repeated twice. One replication gives "
            "**p = 0.042**, the other gives **p = 0.058**. How different is "
            "the evidence from these two replications?"
        ),
        "q4": (
            "Another group runs the study and gets **p = 0.09**. They "
            "conclude: *\"p = 0.09, so the method does not work.\"* "
            "Evaluate this claim."
        ),
    },
}

Q3_SCALE = (
    "1 = almost the same evidence",
    "2",
    "3",
    "4",
    "5 = completely different evidence",
)

POST_ONLY = {
    "q5": (
        "A news headline says: *\"Study proves green tea lowers stress: "
        "only a 4% chance the result is a fluke (p = 0.04).\"* "
        "What, if anything, is wrong with this headline?"
    ),
    "s1": "The feedback I got was about what I specifically wrote.",
    "s1_scale": ("1 = strongly disagree", "2", "3", "4",
                 "5 = strongly agree"),
    "s2": "How much mental effort did the lesson take?",
    "s2_scale": ("1 = very little", "2", "3", "4", "5 = very much"),
    "o1": (
        "What, if anything, helped you understand what a p-value means? "
        "Describe a specific moment."
    ),
    "o2": "Where, if anywhere, did you get stuck? What did you do next?",
    "o3": (
        "When you revised one of your answers, what made you change it "
        "(or keep it)?"
    ),
}

CLOSING_SENTENCE = (
    "One last thought: if you run many tests on data where H₀ is true, "
    "about 1 in 20 of them will still come out \"significant\" at "
    "α = 0.05, just by chance."
)


def study_data():
    """Consent, screening and test answers for this participant."""
    return st.session_state.setdefault("study_data", {})


def now_iso():
    """Current time in ISO format (for the logs)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def render_consent_step():
    """Information sheet, consent and screening question."""
    st.title("Welcome")
    st.markdown(INFO_SHEET)

    data = study_data()
    if "consent" in data:
        st.success("Thank you, your consent has been recorded.")
        if st.button("Continue →", type="primary", width="stretch"):
            st.session_state.step += 1
            st.rerun()
        return

    with st.form("consent_form"):
        agreed = st.checkbox(CONSENT_LABEL)
        screening = st.radio(SCREENING_QUESTION, ("Yes", "No"), index=None,
                             horizontal=True)
        submitted = st.form_submit_button("Start", type="primary")

    if submitted:
        if not agreed:
            st.warning("Please tick the box to agree before starting.")
        elif screening is None:
            st.warning("Please answer the question above.")
        else:
            data["consent"] = {"agreed": True, "time": now_iso()}
            data["screening"] = screening
            st.session_state.step += 1
            st.rerun()


def scale_value(label):
    """'5 = completely different evidence' → 5."""
    return int(str(label)[0])


def render_test(test_name, form_letter, post_only):
    """One test (pre or post): a single form, locked once submitted."""
    form = TEST_FORMS[form_letter]
    data = study_data()

    if test_name in data:
        st.success("Your answers have been submitted.")
        if st.button("Continue →", type="primary", width="stretch",
                     key=f"{test_name}_continue"):
            st.session_state.step += 1
            st.rerun()
        return

    st.info(TEST_RULES)

    with st.form(f"{test_name}_form"):
        st.markdown(form["scenario"])

        st.markdown(f"**Question 1.** {form['q1']}")
        q1 = st.text_area("Your answer", key=f"{test_name}_q1",
                          label_visibility="collapsed")

        st.markdown("**Question 2.** True or false?")
        q2 = [
            st.radio(statement, TRUE_FALSE, index=None, horizontal=True,
                     key=f"{test_name}_q2_{i}")
            for i, (statement, _) in enumerate(form["q2"])
        ]

        st.markdown(f"**Question 3.** {form['q3']}")
        q3_rating = st.radio("Rating", Q3_SCALE, index=None,
                             horizontal=True, key=f"{test_name}_q3_rating")
        q3_why = st.text_area("Explain your rating",
                              key=f"{test_name}_q3_why")

        st.markdown(f"**Question 4.** {form['q4']}")
        q4 = st.text_area("Your answer", key=f"{test_name}_q4",
                          label_visibility="collapsed")

        extra = {}
        if post_only:
            st.divider()
            st.markdown(f"**Question 5.** {POST_ONLY['q5']}")
            extra["q5"] = st.text_area("Your answer", key="post_q5",
                                       label_visibility="collapsed")

            st.divider()
            st.markdown("**About the lesson**")
            extra["s1"] = st.radio(POST_ONLY["s1"], POST_ONLY["s1_scale"],
                                   index=None, horizontal=True,
                                   key="post_s1")
            extra["s2"] = st.radio(POST_ONLY["s2"], POST_ONLY["s2_scale"],
                                   index=None, horizontal=True,
                                   key="post_s2")
            for key in ("o1", "o2", "o3"):
                extra[key] = st.text_area(POST_ONLY[key],
                                          key=f"post_{key}")

        submitted = st.form_submit_button("Submit answers", type="primary")

    if not submitted:
        return

    texts = [q1, q3_why, q4] + [
        extra[key] for key in ("q5", "o1", "o2", "o3") if key in extra
    ]
    choices = q2 + [q3_rating] + [
        extra[key] for key in ("s1", "s2") if key in extra
    ]
    if any(not text.strip() for text in texts) or None in choices:
        st.warning("Please answer every question before submitting.")
        return

    answers = {
        "form": form_letter,
        "time": now_iso(),
        "q1": q1.strip(),
        "q2": q2,
        "q2_score": sum(
            answer == correct
            for answer, (_, correct) in zip(q2, form["q2"])
        ),
        "q3_rating": scale_value(q3_rating),
        "q3_why": q3_why.strip(),
        "q4": q4.strip(),
    }
    if post_only:
        answers.update({
            "q5": extra["q5"].strip(),
            "s1": scale_value(extra["s1"]),
            "s2": scale_value(extra["s2"]),
            "o1": extra["o1"].strip(),
            "o2": extra["o2"].strip(),
            "o3": extra["o3"].strip(),
        })
    data[test_name] = answers
    st.session_state.step += 1
    st.rerun()


def render_pretest_step():
    """Pre-test: first form of the participant's order (AB → A)."""
    st.title("Before the lesson: short test")
    render_test("pretest", st.session_state.study["form_order"][0],
                post_only=False)


def render_posttest_step():
    """Post-test: the other form, plus Q5, S1, S2 and O1-O3."""
    st.title("After the lesson: final questions")
    render_test("posttest", st.session_state.study["form_order"][1],
                post_only=True)


def render_end_step():
    """Thank-you screen with the closing sentence (replaces Step 9)."""
    st.title("Thank you!")
    st.write("You have finished the lesson and the tests.")
    st.info(CLOSING_SENTENCE)
    st.caption(
        f"Participant code: {st.session_state.study['participant_id']}"
    )


@contextmanager
def learning_section(title, completion_key, summary):
    """Keep feedback open until the learner opens a subsequent panel."""
    panel_key = f"learning_{st.session_state.step}_{completion_key}_{title}"
    sections = st.session_state.learning_sections
    waiting_for_next = any(
        st.session_state.get(section["completion_key"], False)
        and st.session_state.get(section["panel_key"], False)
        for section in sections
    )
    sections.append({"panel_key": panel_key, "completion_key": completion_key})
    # Keep the default stable so rerenders preserve user-controlled open state.
    defaults = st.session_state.setdefault("learning_panel_defaults", {})
    expanded = defaults.setdefault(panel_key, not waiting_for_next)
    with st.expander(
        title,
        key=panel_key,
        expanded=expanded,
        icon=":material/menu_book:",
        on_change=open_learning_section,
        args=(panel_key,),
    ):
        yield
        if st.session_state.get(completion_key, False):
            st.caption(summary)


def open_learning_section(panel_key):
    """Opening a panel folds completed panels before it, preserving answers."""
    if not st.session_state.get(panel_key, False):
        return
    for section in st.session_state.learning_sections:
        if section["panel_key"] == panel_key:
            break
        if st.session_state.get(section["completion_key"], False):
            st.session_state[section["panel_key"]] = False


def reset_keys(*keys):
    """
    Reset session state values when the learner changes an answer.
    This prevents old feedback from remaining visible.
    """
    for key in keys:
        st.session_state[key] = False


def normal_cdf(value):
    return 0.5 * (1 + erf(value / sqrt(2)))


def coin_test_results(heads, sample_size, alpha=0.05):
    """Calculate the lesson's two-sided normal approximation under H₀."""
    proportion = heads / sample_size
    standard_error = sqrt(0.25 / sample_size)
    z = (proportion - 0.5) / standard_error
    p_value = erfc(abs(z) / sqrt(2))
    return proportion, standard_error, z, p_value, p_value <= alpha


def resize_coin_sample(prefix):
    """Preserve the observed proportion when the sample size changes."""
    previous_n = st.session_state[f"{prefix}_previous_n"]
    new_n = st.session_state[f"{prefix}_n"]
    proportion = st.session_state.get(
        f"{prefix}_heads", round(previous_n * 0.6)) / previous_n
    st.session_state[f"{prefix}_heads"] = round(proportion * new_n)
    st.session_state[f"{prefix}_previous_n"] = new_n


def coin_distribution_chart(z, alpha=None):
    """Draw the normal curve with observed tails and optional α cutoffs."""
    extent = max(4.0, abs(z) + 0.5)
    critical = NormalDist().inv_cdf(1 - alpha / 2) if alpha is not None else None
    boundaries = [-abs(z), abs(z)]
    if critical is not None:
        boundaries += [-critical, critical]
    x = np.unique(np.concatenate(
        [np.linspace(-extent, extent, 1000), boundaries]))
    frame = pd.DataFrame({
        "Z-score": x,
        "Density": np.exp(-0.5 * x**2) / sqrt(2 * np.pi),
        "side": np.where(x < 0, "left", "right"),
    })
    base = alt.Chart(frame).encode(
        x=alt.X("Z-score:Q", scale=alt.Scale(domain=[-extent, extent])),
        y=alt.Y("Density:Q", scale=alt.Scale(domain=[0, 0.42])),
        tooltip=[alt.Tooltip("Z-score:Q", format=".2f"),
                 alt.Tooltip("Density:Q", format=".4f")],
    )
    layers = []
    if critical is not None:
        rejection = (alt.datum["Z-score"] <= -
                     critical) | (alt.datum["Z-score"] >= critical)
        layers.append(base.transform_filter(rejection).mark_area(
            color="#A78BFA", opacity=0.3).encode(detail="side:N"))
        tails = (alt.datum["Z-score"] <= -abs(z)
                 ) | (alt.datum["Z-score"] >= abs(z))
        layers.append(base.transform_filter(tails).mark_area(
            color="#FBBF24", opacity=0.55).encode(detail="side:N"))
    layers.append(base.mark_line(color="#60A5FA", strokeWidth=3))
    layers.append(alt.Chart(pd.DataFrame({"Z-score": [0]})).mark_rule(
        color="#94A3B8", strokeDash=[4, 4]).encode(x="Z-score:Q"))
    if critical is not None:
        layers.append(alt.Chart(pd.DataFrame({"Z-score": [-critical, critical]})).mark_rule(
            color="#A78BFA", strokeDash=[6, 4]).encode(x="Z-score:Q"))
    markers = [z] if alpha is None else sorted(set([-abs(z), abs(z)]))
    layers.append(alt.Chart(pd.DataFrame({"Z-score": markers})).mark_rule(
        color="#FBBF24", strokeWidth=2).encode(x="Z-score:Q", tooltip=alt.Tooltip("Z-score:Q", format=".2f")))
    return alt.layer(*layers).properties(height=300)


def render_alpha_explanation():
    """Explain how α is chosen and what its error rate means."""
    st.subheader("Choosing the significance level α")
    st.markdown(
        "In hypothesis testing, the **significance level is chosen before "
        "examining the results**. The most commonly used value is **α = 0.05**."
    )
    st.markdown(
        "This means we are willing to accept a **5% probability of a Type I "
        "error**: rejecting H₀ when it is actually true. If H₀ is true, about "
        "5% of repeated tests would reject it at this level. We will explore "
        "Type I errors further in **Step 6**."
    )
    st.markdown(
        "**Common significance levels**\n\n"
        "| Significance level | Risk of a Type I error when H₀ is true |\n"
        "|---|---|\n"
        "| α = 0.10 | 10% risk |\n"
        "| α = 0.05 | 5% risk · most common |\n"
        "| α = 0.01 | 1% risk · stricter |"
    )
    st.info(
        "Try α = 0.10, 0.05, and 0.01 and watch the **purple rejection "
        "regions** change. A smaller α makes them narrower, requiring "
        "stronger evidence to reject H₀. Then change the number of Heads "
        "to change the p-value and the **gold shaded tails**. Changing α "
        "here is an exploration exercise; in a real test, choose it in advance."
    )


def render_coin_explorer(prefix, show_inference=False, explain_alpha=False):
    """Render a reusable experiment without changing the worked examples."""
    st.caption(
        "Experiment freely here. Lesson questions use the original 60 Heads out of 100.")
    sample_size = st.slider(
        "Sample size (number of flips)", 20, 1000, 100, step=20,
        key=f"{prefix}_n", on_change=resize_coin_sample, args=(prefix,),
    )
    st.session_state[f"{prefix}_previous_n"] = sample_size
    heads_key = f"{prefix}_heads"
    if heads_key not in st.session_state:
        st.session_state[heads_key] = round(sample_size * 0.6)
    heads = st.slider(
        f"Number of Heads out of {sample_size}", 0, sample_size, key=heads_key)
    if show_inference and explain_alpha:
        render_alpha_explanation()
    alpha = st.slider(
        "Significance level α", 0.01, 0.10, 0.05, step=0.01,
        key=f"{prefix}_alpha", format="%.2f",
    ) if show_inference else 0.05
    proportion, se, z, p_value, reject = coin_test_results(
        heads, sample_size, alpha)
    columns = st.columns(3)
    columns[0].metric("Observed proportion", f"{proportion:.3f}")
    columns[1].metric("Standard error", f"{se:.4f}")
    columns[2].metric("Z-score", f"{z:.2f}")
    st.altair_chart(
        coin_distribution_chart(z, alpha if show_inference else None),
        width="stretch",
        alt=f"Normal curve under H₀ with the observed result at Z = {z:.2f}." + (
            f" Shaded tails show p = {p_value:.4g}; purple boundaries show α = {alpha:.2f}."
            if show_inference else ""
        ),
    )
    if show_inference:
        st.caption(
            "Gold: outcomes at least as extreme as your result (p-value). Purple: rejection regions set by α. Dashed gray: H₀ at Z = 0.")
        metrics = st.columns(2)
        metrics[0].metric(
            "Two-sided p-value", f"{p_value:.4f}" if p_value >= 0.0001 else f"{p_value:.2e}")
        metrics[1].metric("Significance level α", f"{alpha:.2f}")
        if reject:
            st.success(
                f"Reject H₀: p = {p_value:.4g} ≤ α = {alpha:.2f}. This sample provides evidence against a fair coin.")
        else:
            st.info(
                f"Fail to reject H₀: p = {p_value:.4g} > α = {alpha:.2f}. This sample does not provide enough evidence against a fair coin.")
        st.caption("Changing α moves the decision boundary; it does not change the p-value. Values use the normal approximation taught in this activity.")
    else:
        st.caption("Gold: your observed Z-score. Dashed gray: H₀ at Z = 0. Increase the sample size to see how the same proportion moves farther from zero.")


def navigation_buttons(show_continue=True):
    """
    Standard navigation shown at the bottom of Steps 1–8.
    """

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "← Previous",
            key=f"previous_{st.session_state.step}",
            width="stretch"
        ):
            st.session_state.step -= 1
            st.rerun()

    with col2:
        if st.button(
            "↺ Start Over",
            key=f"restart_{st.session_state.step}",
            width="stretch"
        ):
            restart_activity()
            st.rerun()

    with col3:
        if st.button(
            "Continue →",
            key=f"continue_{st.session_state.step}",
            type="primary" if show_continue else "secondary",
            disabled=not show_continue,
            width="stretch"
        ):
            st.session_state.step += 1
            st.rerun()


def initialize_session_state():
    """Set the initial step for a new activity session."""
    if "step" not in st.session_state:
        st.session_state.step = 0


def render_progress():
    """Show named lesson steps, highlighting the learner's position.

    Not shown on the study screens (consent, tests, end).
    """
    current_step = st.session_state.step
    if STEP_NUMBERS[current_step] is None:
        return

    lesson = [
        index for index, number in enumerate(STEP_NUMBERS)
        if number is not None
    ]
    position = lesson.index(current_step)
    current_label, _ = STEP_RENDERERS[current_step]
    progress = min(position / (len(lesson) - 1), 1.0)
    progress_text = (
        "Introduction · Your learning journey"
        if STEP_NUMBERS[current_step] == ""
        else (
            f"Step {STEP_NUMBERS[current_step]} of {STEP_NUMBERS[lesson[-1]]}"
            f" · {current_label}"
        )
    )
    st.progress(progress, text=progress_text)

    with st.container(horizontal=True, gap="xsmall"):
        for index in lesson:
            label, _ = STEP_RENDERERS[index]
            number = STEP_NUMBERS[index]
            step_label = f"{number}. {label}" if number else label
            if index < current_step:
                st.badge(step_label, icon=":material/check:", color="green")
            elif index == current_step:
                st.badge(step_label, icon=":material/arrow_forward:",
                         color="blue")
            else:
                st.badge(step_label, color="gray")


def render_introduction():
    """Render step 0 and its answer checks."""

    st.title("Hypothesis Testing Activity")

    st.subheader(
        "Can you determine whether a mysterious coin is actually fair?"
    )

    st.write(
        """
        In this activity, you'll investigate a coin using statistical evidence.

        Along the way, you'll learn how **hypothesis testing** works starting from
        from understanding samples all the way to making a statistical decision.
        """
    )

    st.divider()

    st.header("🪙 The Fair Coin Investigation")

    st.write(
        """
        A manufacturer claims that their coin is perfectly fair.

        A fair coin should have a **50% chance of landing Heads**.

        You decide to investigate.
        """
    )

    st.info(
        """
        You flip the coin **100 times**.

        🟢 Heads: **60**

        ⚪ Tails: **40**
        """
    )

    answer = st.radio(
        "What is your first intuition?",
        [
            "The coin is probably fair",
            "The coin might be biased",
            "I'm not sure yet"
        ],
        index=None,
        key="intuition_question"
    )

    if answer is not None:

        st.success(
            "Good! But intuition alone isn't enough. "
            "Let's investigate using statistics."
        )

        st.divider()

        if st.button(
            "Start Investigation →",
            type="primary",
            width="stretch"
        ):
            st.session_state.step += 1
            st.rerun()


def render_data_step():
    """Render step 1 and its answer checks."""

    st.title("Step 1: Understanding the Data")

    st.write(
        """
        Before we test whether the coin is fair,
        let's understand what our observations represent.
        """
    )

    # --------------------------------------------------------
    # Population vs Sample
    # --------------------------------------------------------

    with learning_section('Population vs sample', 'correct_step1', 'The 100 flips are a sample.'):
        st.subheader("Population vs Sample")

        st.info(
            "We flipped the coin **100 times** and observed "
            "**60 Heads and 40 Tails**."
        )

        def reset_sample_answer():
            st.session_state.sample_checked = False
            st.session_state.correct_step1 = False
            st.session_state.step1_complete = False

        question = st.radio(
            "Those 100 flips are:",
            [
                "The population",
                "A sample",
                "A parameter",
                "A hypothesis"
            ],
            index=None,
            key="sample_question",
            on_change=reset_sample_answer
        )

        if st.button("Check Answer", key="check_sample"):

            if question is not None:

                st.session_state.sample_checked = True

                st.session_state.correct_step1 = (
                    question == "A sample"
                )

        if st.session_state.get("sample_checked"):

            if st.session_state.get("correct_step1"):

                st.success(
                    "✅ Correct! The 100 flips are a **sample** from all "
                    "the possible coin flips we could observe."
                )

            else:

                st.error(
                    "Not quite. Think about whether these 100 flips "
                    "represent every possible outcome or only the "
                    "observations we collected."
                )

    # --------------------------------------------------------
    # Reveal remaining material only after correct answer
    # --------------------------------------------------------

    if st.session_state.get("correct_step1"):

        # ----------------------------------------------------
        # Parameter vs Statistic
        # ----------------------------------------------------

        with learning_section('Parameter vs statistic', 'random_n', 'π is the parameter; the observed proportion is a statistic.'):
            st.divider()

            st.subheader("Parameter vs Statistic")

            st.write(
                """
                The true probability that the coin lands on Heads is a
                **population parameter**:
                """
            )

            st.latex(
                r"\pi = \text{true probability of Heads}"
            )

            st.write(
                """
                But we do not know the true value of $\\pi$.

                What we *do* know is what happened in our sample:
                **60 Heads out of 100 flips**.
                """
            )

            st.latex(r"\hat{p} = \frac{60}{100} = 0.60")

            st.markdown(
                r"""
                The observed proportion $\hat{p}$ is a **statistic**.

                We use a statistic calculated from our sample to learn
                something about an unknown population parameter.
                """
            )

        # ----------------------------------------------------
        # Interactive sampling experiment
        # ----------------------------------------------------

        with learning_section('Run a sampling experiment', 'random_n', 'A fair coin can produce different sample proportions.'):
            st.divider()

            st.subheader("🪙 Random samples naturally vary")

            st.write(
                """
                Even if the coin is perfectly fair, we would **not**
                expect every experiment to produce exactly half Heads
                and half Tails.

                Try it yourself.
                """
            )

            n_flips = st.slider(
                "Choose the number of times to flip the fair coin:",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                key="sample_size"
            )

            if (
                "random_n" in st.session_state
                and st.session_state.random_n != n_flips
            ):
                st.caption(
                    "👆 You changed the sample size. "
                    "Flip again to run the new experiment."
                )

            if st.button(
                "🪙 Flip the coin",
                key="flip_coin"
            ):

                heads = np.random.binomial(
                    n_flips,
                    0.5
                )

                tails = n_flips - heads

                st.session_state.random_heads = heads
                st.session_state.random_tails = tails
                st.session_state.random_n = n_flips

        if "random_heads" in st.session_state:

            heads = st.session_state.random_heads
            tails = st.session_state.random_tails
            n = st.session_state.random_n

            p_hat = heads / n

            se = np.sqrt(
                0.5 * (1 - 0.5) / n
            )

            with learning_section('Your random sample', 'step1_complete', 'Random samples naturally vary.'):
                st.write("### Your random sample")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Heads",
                        heads
                    )

                with col2:
                    st.metric(
                        "Tails",
                        tails
                    )

                with col3:
                    st.metric(
                        "Observed proportion",
                        f"{p_hat:.3f}"
                    )

                st.write(
                    f"""
                    This experiment produced **{heads} Heads**
                    and **{tails} Tails**.

                    The observed proportion was **{p_hat:.3f}**
                    instead of exactly **0.500**.
                    """
                )

                st.info(
                    """
                    Even though we simulated a perfectly fair coin,
                    the sample usually does not produce exactly 50% Heads.

                    **Random samples naturally vary.**
                    """
                )

            # ------------------------------------------------
            # Sampling distribution and SE
            # ------------------------------------------------

            with learning_section('Sampling distribution and standard error', 'step1_complete', 'Larger samples have less sampling variability.'):
                st.divider()

                st.subheader(
                    "Sampling Distribution & Standard Error"
                )

                st.write(
                    """
                    Imagine repeating this experiment many times.

                    Each experiment would produce a slightly different
                    observed proportion \(\hat{p}\).

                    The distribution of all these sample proportions
                    is called the **sampling distribution**.
                    """
                )

                st.write(
                    """
                    The **standard error** tells us how much these sample
                    proportions typically vary.
                    """
                )

                st.write(
                    "For a proportion:"
                )

                st.latex(
                    r"SE = \sqrt{\frac{\pi(1-\pi)}{n}}"
                )

                st.write(
                    """
                    Since we are simulating a **fair coin**, we use:
                    """
                )

                st.latex(
                    r"\pi = 0.5"
                )

                st.write(
                    f"For your chosen sample size, \(n={n}\):"
                )

                st.latex(
                    rf"SE = "
                    rf"\sqrt{{\frac{{0.5(1-0.5)}}{{{n}}}}}"
                    rf" = {se:.4f}"
                )

                st.metric(
                    "Standard Error",
                    f"{se:.4f}"
                )

                st.caption(
                    "Try changing the sample size and flipping again. "
                    "Pay attention to what happens to the standard error."
                )

            # ------------------------------------------------
            # Quick check
            # ------------------------------------------------

            with learning_section('Check your understanding', 'step1_complete', 'Larger n means smaller standard error.'):
                st.divider()

                st.subheader("Quick check")

                def reset_se_question():
                    st.session_state.se_checked = False
                    st.session_state.step1_complete = False

                se_question = st.radio(
                    "As the sample size increases, what happens to the standard error?",
                    [
                        "It increases",
                        "It decreases",
                        "It stays the same"
                    ],
                    index=None,
                    key="se_question",
                    on_change=reset_se_question
                )

                if st.button(
                    "Check",
                    key="check_se"
                ):

                    if se_question is not None:

                        st.session_state.se_checked = True

                        st.session_state.step1_complete = (
                            se_question == "It decreases"
                        )

                if st.session_state.get("se_checked"):

                    if st.session_state.get(
                        "step1_complete"
                    ):

                        st.success(
                            "✅ Correct! As the sample size increases, "
                            "the standard error decreases."
                        )

                        st.info(
                            """
                            Larger samples tend to give more stable estimates.

                            **Larger n → smaller SE → less sampling variability**
                            """
                        )

                    else:

                        st.error(
                            "Not quite. Try comparing the standard error "
                            "for a small sample with the standard error "
                            "for a much larger sample."
                        )

    navigation_buttons(
        show_continue=st.session_state.get(
            "step1_complete",
            False
        )
    )


def render_hypotheses_step():
    """Render step 2 and its answer checks."""

    st.title("Step 2: Form Your Hypotheses")

    st.write(
        """
        The manufacturer claims that the coin is fair.

        A fair coin has a probability of Heads equal to:
        """
    )

    st.latex(
        r"\pi = 0.5"
    )

    # --------------------------------------------------------
    # H0
    # --------------------------------------------------------

    with learning_section('Null hypothesis', 'h0_correct', 'H₀: π = 0.5.'):
        st.subheader("Choose the null hypothesis")

        def reset_h0():
            reset_keys(
                "h0_checked",
                "h0_correct",
                "h1_checked",
                "h1_correct",
                "coffee_h1_checked",
                "coffee_h1_correct"
            )

        h0 = st.radio(
            "What should H₀ be?",
            [
                "π = 0.5",
                "π ≠ 0.5",
                "π > 0.5",
                "π < 0.5"
            ],
            index=None,
            key="h0_question",
            on_change=reset_h0
        )

        if st.button(
            "Check H₀",
            key="check_h0"
        ):

            if h0 is not None:

                st.session_state.h0_checked = True
                st.session_state.h0_correct = (
                    h0 == "π = 0.5"
                )

        if st.session_state.get("h0_checked"):

            if st.session_state.get("h0_correct"):

                st.success(
                    "✅ Correct! H₀ represents the default assumption: "
                    "the coin is fair."
                )

            else:

                st.error(
                    "Not quite. H₀ should represent the manufacturer's "
                    "claim that the coin is fair."
                )

    # --------------------------------------------------------
    # H1
    # --------------------------------------------------------

    if st.session_state.get("h0_correct"):

        with learning_section('Alternative hypothesis', 'h1_correct', 'H₁: π ≠ 0.5.'):
            st.subheader(
                "Now choose the alternative hypothesis"
            )

            def reset_h1():
                reset_keys(
                    "h1_checked",
                    "h1_correct",
                    "coffee_h1_checked",
                    "coffee_h1_correct"
                )

            h1 = st.radio(
                "What should H₁ be?",
                [
                    "π = 0.5",
                    "π ≠ 0.5",
                    "π > 0.5",
                    "π < 0.5"
                ],
                index=None,
                key="h1_question",
                on_change=reset_h1
            )

            if st.button(
                "Check H₁",
                key="check_h1"
            ):

                if h1 is not None:

                    st.session_state.h1_checked = True

                    st.session_state.h1_correct = (
                        h1 == "π ≠ 0.5"
                    )

            if st.session_state.get("h1_checked"):

                if st.session_state.get(
                    "h1_correct"
                ):

                    st.success(
                        "✅ Correct! We want to detect bias "
                        "in either direction."
                    )

                else:

                    st.error(
                        "Not quite. We are asking whether the coin "
                        "is biased toward Heads **or** Tails."
                    )

    # --------------------------------------------------------
    # Alternative directions
    # --------------------------------------------------------

    if (
        st.session_state.get("h0_correct")
        and st.session_state.get("h1_correct")
    ):

        with learning_section('Apply hypotheses to a new example', 'coffee_h1_correct', 'Suspecting less coffee means H₁: μ < 250.'):
            st.info(
                """
                Our hypotheses are:

                **H₀: π = 0.5**

                **H₁: π ≠ 0.5**

                H₀ says the coin is fair.

                H₁ says the coin is biased in either direction.
                """
            )

            st.divider()

            st.subheader(
                "The direction of H₁ depends on the question"
            )

            st.markdown(
                """
                **What are you trying to detect?**

                - Increase → $>$
                - Decrease → $<$
                - Any difference → $\\neq$
                """
            )

            st.subheader("Try another example")

            st.info(
                """
                A coffee machine should pour **250 ml on average**.

                You suspect that it actually pours **less**.
                """
            )

            def reset_coffee_h1():
                reset_keys(
                    "coffee_h1_checked",
                    "coffee_h1_correct"
                )

            coffee_h1 = st.radio(
                "Which alternative hypothesis is correct?",
                [
                    "H₁: μ = 250",
                    "H₁: μ ≠ 250",
                    "H₁: μ < 250",
                    "H₁: μ > 250"
                ],
                index=None,
                key="coffee_h1",
                on_change=reset_coffee_h1
            )

            if st.button(
                "Check coffee example",
                key="check_coffee_h1"
            ):

                if coffee_h1 is not None:

                    st.session_state.coffee_h1_checked = True

                    st.session_state.coffee_h1_correct = (
                        coffee_h1 == "H₁: μ < 250"
                    )

            if st.session_state.get(
                "coffee_h1_checked"
            ):

                if st.session_state.get(
                    "coffee_h1_correct"
                ):

                    st.success(
                        "✅ Correct! The question asks whether "
                        "the machine pours **less**, so H₁ uses <."
                    )

                    st.info(
                        """
                        **Takeaway**

                        H₀ = default/reference assumption

                        H₁ = effect or difference we are looking for
                        """
                    )

                else:

                    st.error(
                        "Not quite. Focus on the word **less** "
                        "in the question."
                    )

    navigation_buttons(
        show_continue=st.session_state.get(
            "coffee_h1_correct",
            False
        )
    )


def render_test_statistic_step():
    """Render Step 3: Explore and understand the Z-score."""

    st.title("Step 3: How far is our result from H₀?")

    # --------------------------------------------------------
    # Introduction: Why do we need a Z-score?
    # --------------------------------------------------------

    st.write(
        """
        We flipped our coin **100 times** and observed **60 Heads**.

        Under our null hypothesis, the coin is fair:
        """
    )

    st.latex(r"H_0: \pi = 0.5")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Expected proportion", "0.50")

    with col2:
        st.metric("Observed proportion", "0.60")

    st.write(
        """
        The difference is **0.10 (10 percentage points)**.

        But is this a large difference, or could it simply
        happen because of random sampling variability?

        To answer this, we need to consider how much
        our samples naturally vary.
        """
    )

    # --------------------------------------------------------
    # 1. Interactive exploration
    # --------------------------------------------------------

    with learning_section(
        "Explore the Z-score",
        "z_exploration_correct",
        "The farther the observation is from H₀, the larger |Z| becomes."
    ):

        st.subheader("Explore what happens")

        st.write(
            """
            Change the number of Heads or the sample size.

            With **100 flips**, try **50, 55, 60, and 65 Heads**.

            Watch how the Z-score and its position on the
            distribution change. Changing the sample size keeps
            approximately the same observed proportion.
            """
        )

        render_coin_explorer("z_experiment")

        # ----------------------------------------------------
        # 2. Prediction question
        # ----------------------------------------------------

        st.divider()
        st.subheader("What did you notice?")

        def reset_z_exploration():
            reset_keys(
                "z_exploration_checked",
                "z_exploration_correct"
            )

        exploration_answer = st.radio(
            "What happens as the observed proportion moves farther from 0.50?",
            [
                "The Z-score always becomes positive",
                "The absolute Z-score increases",
                "The Z-score moves closer to zero"
            ],
            index=None,
            key="z_exploration_answer",
            on_change=reset_z_exploration
        )

        if st.button(
            "Check observation",
            key="check_z_exploration"
        ):
            if exploration_answer is not None:
                st.session_state.z_exploration_checked = True
                st.session_state.z_exploration_correct = (
                    exploration_answer == "The absolute Z-score increases"
                )

        if st.session_state.get("z_exploration_checked"):

            if st.session_state.get("z_exploration_correct"):

                st.success(
                    "✅ Correct! The farther the observation is "
                    "from what H₀ predicts, the larger its "
                    "absolute Z-score becomes."
                )

                st.info(
                    "The sign tells us the **direction**, "
                    "while the absolute value tells us the **distance**."
                )

            else:
                st.error(
                    "Not quite. Try moving the slider both "
                    "above and below 50 Heads. "
                    "Pay attention to the distance from zero."
                )

    # --------------------------------------------------------
    # 3. Explain the Z-score
    # --------------------------------------------------------

    if st.session_state.get("z_exploration_correct"):

        with learning_section(
            "Understand the Z-score",
            "z_explanation_complete",
            "Z measures distance from H₀ in standard errors."
        ):

            st.subheader("What is a Z-score?")

            st.write(
                """
                The **Z-score** measures how far our observed
                result is from what H₀ predicts, taking normal
                sampling variability into account.

                It expresses this distance in units of
                **standard errors**.
                """
            )

            st.latex(
                r"Z = \frac{\text{Observed} - "
                r"\text{Expected under }H_0}"
                r"{\text{Standard Error}}"
            )

            st.write(
                """
                For our original experiment, we observed
                **60 Heads out of 100**.
                """
            )

            st.latex(
                r"SE = \sqrt{\frac{0.5(1-0.5)}{100}} = 0.05"
            )

            st.latex(
                r"Z = \frac{0.60-0.50}{0.05} = 2"
            )

            st.info(
                """
                **Z = 2** means our observed proportion
                is **2 standard errors above** what H₀ predicts.

                The Z-score has **no unit**.
                """
            )

            st.subheader("Understanding the Z-score scale")

            st.markdown("""
- **Z = 0** → exactly what H₀ predicts
- **|Z| ≈ 1** → common
- **|Z| ≈ 2** → unusual
- **|Z| ≈ 3** → very unusual
""")

            st.caption(
                "These are intuition guidelines, not formal decision thresholds."
            )

            st.write("### Positive vs Negative Z")

            st.write(
                """
                - **Positive Z:** the observation is above
                  the value predicted by H₀.
                - **Negative Z:** the observation is below
                  the value predicted by H₀.

                The larger **|Z|** is, the farther our
                observation is from the null hypothesis.
                """
            )

            if st.button(
                "I understand. Let's continue to practice →",
                key="z_explanation_done"
            ):
                st.session_state.z_explanation_complete = True

    # --------------------------------------------------------
    # 4. Independent transfer exercise
    # --------------------------------------------------------

    if st.session_state.get("z_explanation_complete"):

        with learning_section(
            "Try a Z-score calculation",
            "z_transfer_correct",
            "(490 − 500) / 5 = −2."
        ):

            st.subheader("Quick transfer check")

            st.info(
                """
                Suppose:

                **H₀: μ = 500**

                Observed sample mean: **490**

                Standard error: **5**
                """
            )

            st.write(
                "Try calculating the Z-score on your own."
            )

            with st.expander("Need a hint?"):
                st.write(
                    "Subtract the value predicted by H₀ "
                    "from the observed value, then divide "
                    "by the standard error."
                )

                st.latex(
                    r"Z = \frac{\text{Observed} - "
                    r"\text{Expected}}{SE}"
                )

            def reset_z_transfer():
                reset_keys(
                    "z_transfer_checked",
                    "z_transfer_correct"
                )

            z_transfer = st.radio(
                "What is the Z-score?",
                [
                    "Z = -2",
                    "Z = -1",
                    "Z = 1",
                    "Z = 2"
                ],
                index=None,
                key="z_transfer",
                on_change=reset_z_transfer
            )

            if st.button(
                "Check Z-score",
                key="check_z_transfer"
            ):

                if z_transfer is not None:

                    st.session_state.z_transfer_checked = True

                    st.session_state.z_transfer_correct = (
                        z_transfer == "Z = -2"
                    )

            if st.session_state.get("z_transfer_checked"):

                if st.session_state.get("z_transfer_correct"):

                    st.success(
                        "✅ Correct! The observed mean is "
                        "2 standard errors below H₀."
                    )

                    st.latex(
                        r"Z = \frac{490-500}{5} = -2"
                    )

                else:

                    st.error(
                        "Not quite. Think about the direction "
                        "of the difference, then calculate "
                        "(490 - 500) / 5."
                    )

    # --------------------------------------------------------
    # 5. Transition to the p-value
    # --------------------------------------------------------

    if st.session_state.get("z_transfer_correct"):

        st.divider()

        st.subheader("What's next?")

        st.write(
            """
            We now know that our original coin experiment
            gave **Z = 2**.

            This means our result is 2 standard errors
            away from what H₀ predicts.

            But **how unlikely is a result this extreme
            if the coin is actually fair?**

            That's what we'll investigate next using
            the **p-value**.
            """
        )

    navigation_buttons(
        show_continue=st.session_state.get(
            "z_transfer_correct",
            False
        )
    )


def render_pvalue_step():
    """Render step 4 and its answer checks."""

    st.title(
        "Step 4: How surprising is this result under H₀?"
    )

    st.write(
        """
        In the previous step, we measured how far our result
        was from H₀ using the Z-score.

        Now we ask:

        **If H₀ were true, how likely would it be to observe
        a result this extreme or even more extreme?**
        """
    )

    st.latex(
        r"H_0: \pi = 0.5"
    )

    st.latex(
        r"H_1: \pi \neq 0.5"
    )

    st.info(
        """
        Because H₁ uses **≠**, we care about unusual results
        in **both directions**.

        This means we have a **two-sided test**.
        """
    )

    heads = 60
    p_hat = 0.60
    p0 = 0.5
    n = 100

    se = np.sqrt(
        p0 * (1 - p0) / n
    )

    z = (
        p_hat - p0
    ) / se

    st.subheader("Our observed result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Z-score",
            f"{z:.2f}"
        )

    with col2:
        st.metric(
            "Observed proportion",
            f"{p_hat:.2f}"
        )

    # --------------------------------------------------------
    # Misconception check
    # --------------------------------------------------------

    with learning_section('What does a p-value mean?', 'f1_complete', 'The p-value assumes H₀ is true.'):
        st.subheader(
            "What does the p-value tell us?"
        )

        st.write(
            "For our 60 heads, the two-sided p-value turns out to be "
            "about 0.046. Before we look at where it comes from, "
            "interpret it in your own words."
        )

        st.session_state.f1_complete = render_free_text_activity("F1")

    # --------------------------------------------------------
    # Coin p-value visualization
    # --------------------------------------------------------

    if st.session_state.get(
        "f1_complete"
    ):

        with learning_section('Visualize the two-sided p-value', 'coffee_z_correct', 'The two-sided p-value includes outcomes in both tails.'):
            st.subheader(
                "Visualizing the p-value"
            )

            render_coin_explorer("pvalue_experiment",
                                 show_inference=True, explain_alpha=True)

        # ----------------------------------------------------
        # Coffee machine exercise
        # ----------------------------------------------------

        with learning_section('Calculate the coffee machine Z-score', 'coffee_z_correct', '(244 − 250) / 2 = −3.'):
            st.divider()

            st.subheader(
                "Now try a different example"
            )

            st.info(
                """
                A coffee machine should pour **250 ml on average**.

                You suspect it pours **less**.

                From a sample:

                **x̄ = 244 ml**

                **SE = 2 ml**
                """
            )

            def reset_coffee_z():
                reset_keys(
                    "coffee_z_checked",
                    "coffee_z_correct",
                    "coffee_tail_checked",
                    "coffee_tail_correct",
                    "pvalue_reflection_checked",
                    "pvalue_reflection_correct"
                )

            coffee_z = st.radio(
                "What is the test statistic?",
                [
                    "Z = -1",
                    "Z = -2",
                    "Z = -3",
                    "Z = 3"
                ],
                index=None,
                key="coffee_z",
                on_change=reset_coffee_z
            )

            if st.button(
                "Check Z",
                key="check_coffee_z"
            ):

                if coffee_z is not None:

                    st.session_state.coffee_z_checked = True

                    st.session_state.coffee_z_correct = (
                        coffee_z == "Z = -3"
                    )

            if st.session_state.get(
                "coffee_z_checked"
            ):

                if st.session_state.get(
                    "coffee_z_correct"
                ):

                    st.success(
                        "✅ Correct!"
                    )

                    st.latex(
                        r"Z = \frac{244-250}{2} = -3"
                    )

                else:

                    st.error(
                        "Not quite. Calculate (244 - 250) / 2."
                    )

        # ----------------------------------------------------
        # Tail
        # ----------------------------------------------------

        if st.session_state.get(
            "coffee_z_correct"
        ):

            with learning_section('Choose the correct tail', 'coffee_tail_correct', 'H₁: μ < 250 uses the left tail.'):
                def reset_coffee_tail():
                    reset_keys(
                        "coffee_tail_checked",
                        "coffee_tail_correct",
                        "pvalue_reflection_checked",
                        "pvalue_reflection_correct"
                    )

                tail_answer = st.radio(
                    "Since H₁: μ < 250, which probability gives the p-value?",
                    [
                        "P(Z ≥ -3)",
                        "P(Z ≤ -3)",
                        "2P(Z ≤ -3)",
                        "P(-3 ≤ Z ≤ 3)"
                    ],
                    index=None,
                    key="coffee_tail",
                    on_change=reset_coffee_tail
                )

                if st.button(
                    "Check tail",
                    key="check_coffee_tail"
                ):

                    if tail_answer is not None:

                        st.session_state.coffee_tail_checked = True

                        st.session_state.coffee_tail_correct = (
                            tail_answer == "P(Z ≤ -3)"
                        )

                if st.session_state.get(
                    "coffee_tail_checked"
                ):

                    if st.session_state.get(
                        "coffee_tail_correct"
                    ):

                        st.success(
                            "✅ Correct! Because H₁ uses <, "
                            "this is a left-tailed test."
                        )

                    else:

                        st.error(
                            "Not quite. H₁ uses <, so we look "
                            "toward the left tail."
                        )

        # ----------------------------------------------------
        # Coffee p-value interpretation
        # ----------------------------------------------------

        if st.session_state.get(
            "coffee_tail_correct"
        ):

            with learning_section('Interpret the coffee machine p-value', 'f2_complete', 'Under H₀, a mean this low or lower is very unusual.'):
                st.latex(
                    r"p = P(Z \leq -3) \approx 0.00135"
                )

                st.session_state.f2_complete = render_free_text_activity(
                    "F2"
                )

        # ----------------------------------------------------
        # Final reflection
        # ----------------------------------------------------

        if st.session_state.get(
            "f2_complete"
        ):

            with learning_section('Final reflection', 'pvalue_reflection_correct', 'Farther from H₀ means a smaller p-value.'):
                st.subheader("Final reflection")

                def reset_p_reflection():
                    reset_keys(
                        "pvalue_reflection_checked",
                        "pvalue_reflection_correct"
                    )

                reflection = st.radio(
                    "As the observed result moves farther away from H₀, what happens to the p-value?",
                    [
                        "It increases",
                        "It decreases",
                        "It stays the same"
                    ],
                    index=None,
                    key="pvalue_reflection",
                    on_change=reset_p_reflection
                )

                if st.button(
                    "Check reflection",
                    key="check_pvalue_reflection"
                ):

                    if reflection is not None:

                        st.session_state.pvalue_reflection_checked = True

                        st.session_state.pvalue_reflection_correct = (
                            reflection == "It decreases"
                        )

                if st.session_state.get(
                    "pvalue_reflection_checked"
                ):

                    if st.session_state.get(
                        "pvalue_reflection_correct"
                    ):

                        st.success(
                            "✅ Correct!"
                        )

                        st.latex(
                            r"|Z| \uparrow "
                            r"\Rightarrow "
                            r"p\text{-value} \downarrow"
                        )

                    else:

                        st.error(
                            "Not quite. Think about what happens when "
                            "an observation becomes increasingly unusual."
                        )

    navigation_buttons(
        show_continue=st.session_state.get(
            "pvalue_reflection_correct",
            False
        )
    )


def render_significance_step():
    """Render step 5 and its answer checks."""

    st.title(
        "Step 5: Is the p-value small enough?"
    )

    st.write(
        """
        We obtained:
        """
    )

    p_value = 0.046
    alpha = 0.05

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "p-value",
            f"{p_value:.3f}"
        )

    with col2:
        st.metric(
            "α",
            f"{alpha:.2f}"
        )

    st.write(
        """
        The significance level **α** is the threshold we use
        to decide whether the evidence against H₀ is strong enough.
        """
    )

    st.latex(
        r"p \leq \alpha "
        r"\Rightarrow "
        r"\text{Reject } H_0"
    )

    st.latex(
        r"p > \alpha "
        r"\Rightarrow "
        r"\text{Fail to reject } H_0"
    )

    with st.expander("Experiment: how do p-value and α change the decision?"):
        render_coin_explorer("decision_experiment", show_inference=True)

    # --------------------------------------------------------
    # Coin decision
    # --------------------------------------------------------

    with learning_section('Make the coin decision', 'decision_correct', '0.046 < 0.05: reject H₀.'):
        def reset_decision():
            reset_keys(
                "decision_checked",
                "decision_correct",
                "decision2_checked",
                "decision2_correct"
            )

        decision = st.radio(
            "What should we conclude?",
            [
                "Accept H₀",
                "Reject H₀",
                "Fail to reject H₀",
                "There is not enough information"
            ],
            index=None,
            key="decision_question",
            on_change=reset_decision
        )

        if st.button(
            "Check decision",
            key="check_decision"
        ):

            if decision is not None:

                st.session_state.decision_checked = True

                st.session_state.decision_correct = (
                    decision == "Reject H₀"
                )

        if st.session_state.get(
            "decision_checked"
        ):

            if st.session_state.get(
                "decision_correct"
            ):

                st.success(
                    "✅ Correct! Since p = 0.046 is smaller "
                    "than α = 0.05, we reject H₀."
                )

            else:

                st.error(
                    "Not quite. Compare the p-value directly with α."
                )

    if st.session_state.get(
        "decision_correct"
    ):

        with learning_section('What rejection means', 'decision2_correct', 'Significant evidence is not absolute certainty.'):
            st.latex(
                r"0.046 < 0.05"
            )

            st.success(
                "Reject H₀"
            )

            st.write(
                """
                The data provide **statistically significant evidence**
                against the assumption that the coin is fair.
                """
            )

            st.warning(
                """
                Be careful:

                We should **not** say:

                **"The coin is definitely biased."**

                A better conclusion is:

                **"The data provide sufficient evidence against the
                assumption that the coin is fair."**
                """
            )

        # ----------------------------------------------------
        # Fail-to-reject example
        # ----------------------------------------------------

        with learning_section('Try the opposite case', 'decision2_correct', '0.12 > 0.05: fail to reject H₀.'):
            st.divider()

            st.subheader(
                "What about the opposite case?"
            )

            st.write(
                "Suppose another experiment gives:"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "p-value",
                    "0.12"
                )

            with col2:
                st.metric(
                    "α",
                    "0.05"
                )

            def reset_decision2():
                reset_keys(
                    "decision2_checked",
                    "decision2_correct"
                )

            decision2 = st.radio(
                "What should we conclude now?",
                [
                    "Reject H₀",
                    "Fail to reject H₀",
                    "Accept H₀"
                ],
                index=None,
                key="decision2",
                on_change=reset_decision2
            )

            if st.button(
                "Check second decision",
                key="check_decision2"
            ):

                if decision2 is not None:

                    st.session_state.decision2_checked = True

                    st.session_state.decision2_correct = (
                        decision2 == "Fail to reject H₀"
                    )

            if st.session_state.get(
                "decision2_checked"
            ):

                if st.session_state.get(
                    "decision2_correct"
                ):

                    st.success(
                        "✅ Correct! Since 0.12 > 0.05, "
                        "we fail to reject H₀."
                    )

                    st.warning(
                        """
                        **Failing to reject H₀ does NOT mean that H₀
                        has been proven true.**

                        It only means we do not have strong enough
                        evidence against it.
                        """
                    )

                else:

                    st.error(
                        "Not quite. Compare 0.12 with 0.05."
                    )

    navigation_buttons(
        show_continue=st.session_state.get(
            "decision2_correct",
            False
        )
    )


def render_threshold_step():
    """Render step 5b: is 0.05 a hard line? (free-text prompt F3)."""

    st.title(
        "Step 5b: Is 0.05 a hard line?"
    )

    st.write(
        """
        In Step 5, 60 heads gave p ≈ 0.046, so we rejected H₀ at
        α = 0.05.

        Now imagine the same experiment, but the coin landed heads
        **59** times instead of 60.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("60 heads", "p ≈ 0.046")
        st.caption("0.046 < 0.05 → reject H₀")

    with col2:
        st.metric("59 heads", "p ≈ 0.072")
        st.caption("0.072 > 0.05 → fail to reject H₀")

    st.divider()

    st.session_state.f3_complete = render_free_text_activity("F3")

    navigation_buttons(
        show_continue=st.session_state.get("f3_complete", False)
    )


def render_errors_step():
    """Render step 6 (errors and power) and its answer checks."""

    st.title(
        "Step 6: What if our decision is wrong?"
    )

    st.write(
        """
        Hypothesis testing gives us a decision based on evidence.

        But sometimes that decision can be wrong.
        """
    )

    # --------------------------------------------------------
    # Scenario 1
    # --------------------------------------------------------

    with learning_section('Scenario 1: a false positive', 'type1_correct', 'Rejecting a true H₀ is a Type I error.'):
        st.subheader("Scenario 1")

        st.info(
            """
            Suppose the coin is actually **fair**.

            But our hypothesis test concludes that the coin is **biased**.
            """
        )

        def reset_type1():
            reset_keys(
                "type1_checked",
                "type1_correct",
                "type2_checked",
                "type2_correct",
                "med_error_checked",
                "med_error_correct"
            )

        error1 = st.radio(
            "What happened?",
            [
                "Type I error",
                "Type II error",
                "No error"
            ],
            index=None,
            key="type1_question",
            on_change=reset_type1
        )

        if st.button(
            "Check Scenario 1",
            key="check_type1"
        ):

            if error1 is not None:

                st.session_state.type1_checked = True

                st.session_state.type1_correct = (
                    error1 == "Type I error"
                )

        if st.session_state.get(
            "type1_checked"
        ):

            if st.session_state.get(
                "type1_correct"
            ):

                st.success(
                    "✅ Correct! We rejected H₀ even though "
                    "H₀ was actually true."
                )

            else:

                st.error(
                    "Not quite. Remember that H₀ says the coin is fair."
                )

    if st.session_state.get(
        "type1_correct"
    ):

        with learning_section('Type I error explained', 'type2_correct', 'α controls the probability of a Type I error.'):
            st.write("### Type I Error")

            st.latex(
                r"\text{Type I error} = "
                r"\text{Reject a true } H_0"
            )

            st.write(
                """
                This is also called a **false positive**.
                """
            )

            st.info(
                """
                The probability of making a Type I error
                is controlled by **α**.
                """
            )

        # ----------------------------------------------------
        # Scenario 2
        # ----------------------------------------------------

        with learning_section('Scenario 2: a missed effect', 'type2_correct', 'Failing to reject a false H₀ is a Type II error.'):
            st.divider()

            st.subheader("Scenario 2")

            st.info(
                """
                Now suppose the coin is actually **biased**.

                But our experiment does **not** provide enough evidence
                to reject H₀.
                """
            )

            def reset_type2():
                reset_keys(
                    "type2_checked",
                    "type2_correct",
                    "med_error_checked",
                    "med_error_correct"
                )

            error2 = st.radio(
                "What happened this time?",
                [
                    "Type I error",
                    "Type II error",
                    "No error"
                ],
                index=None,
                key="type2_question",
                on_change=reset_type2
            )

            if st.button(
                "Check Scenario 2",
                key="check_type2"
            ):

                if error2 is not None:

                    st.session_state.type2_checked = True

                    st.session_state.type2_correct = (
                        error2 == "Type II error"
                    )

            if st.session_state.get(
                "type2_checked"
            ):

                if st.session_state.get(
                    "type2_correct"
                ):

                    st.success(
                        "✅ Correct! H₀ was false, "
                        "but we failed to reject it."
                    )

                else:

                    st.error(
                        "Not quite. The coin really is biased, "
                        "but the test failed to detect it."
                    )

    if st.session_state.get(
        "type2_correct"
    ):

        with learning_section('Type II error and summary', 'med_error_correct', 'A false negative misses a real effect.'):
            st.write("### Type II Error")

            st.latex(
                r"\text{Type II error} = "
                r"\text{Fail to reject a false } H_0"
            )

            st.write(
                """
                This is also called a **false negative**.

                A real effect exists, but our test fails to detect it.
                """
            )

            st.subheader("Quick summary")

            st.markdown(
                """
                | Reality | Decision | Result |
                |---|---|---|
                | H₀ is true | Reject H₀ | **Type I error** |
                | H₀ is true | Fail to reject H₀ | Correct decision |
                | H₀ is false | Reject H₀ | Correct decision |
                | H₀ is false | Fail to reject H₀ | **Type II error** |
                """
            )

        # ----------------------------------------------------
        # Transfer example
        # ----------------------------------------------------

        with learning_section('Apply errors to a medication test', 'med_error_correct', 'Detecting an effect that does not exist is a Type I error.'):
            st.divider()

            st.subheader("Transfer example")

            st.info(
                """
                Suppose we test whether a new medication works.

                The medication actually **does not work**,
                but the test concludes that it **does work**.
                """
            )

            def reset_med():
                reset_keys(
                    "med_error_checked",
                    "med_error_correct"
                )

            med_error = st.radio(
                "What type of error is this?",
                [
                    "Type I error",
                    "Type II error"
                ],
                index=None,
                key="med_error",
                on_change=reset_med
            )

            if st.button(
                "Check medication example",
                key="check_med_error"
            ):

                if med_error is not None:

                    st.session_state.med_error_checked = True

                    st.session_state.med_error_correct = (
                        med_error == "Type I error"
                    )

            if st.session_state.get(
                "med_error_checked"
            ):

                if st.session_state.get(
                    "med_error_correct"
                ):

                    st.success(
                        "✅ Correct! We concluded that an effect "
                        "exists when it actually does not."
                    )

                else:

                    st.error(
                        "Not quite. Think about whether this is "
                        "a false positive or a false negative."
                    )

    # --------------------------------------------------------
    # Statistical power (former Step 7, trimmed to one MCQ)
    # --------------------------------------------------------

    if st.session_state.get(
        "med_error_correct"
    ):

        with learning_section('Statistical power', 'power_correct', 'More flips give more power to detect a real bias.'):
            st.divider()

            st.subheader("Can our test detect a real effect?")

            st.write(
                r"""
                The **power** of a test is the probability that it detects
                a real effect: $\text{Power} = 1 - \beta$, where $\beta$
                is the probability of a Type II error.
                """
            )

            st.info(
                """
                Suppose the coin is actually biased: **π = 0.60**.
                """
            )

            def reset_power():
                reset_keys(
                    "power_checked",
                    "power_correct"
                )

            power_answer = st.radio(
                "Which change makes our test more likely to detect "
                "this bias?",
                [
                    "Flip the coin fewer times",
                    "Flip the coin more times",
                    "Nothing: the power of a test cannot be changed"
                ],
                index=None,
                key="power_question",
                on_change=reset_power
            )

            if st.button(
                "Check power",
                key="check_power"
            ):

                if power_answer is not None:

                    st.session_state.power_checked = True

                    st.session_state.power_correct = (
                        power_answer == "Flip the coin more times"
                    )

            if st.session_state.get(
                "power_checked"
            ):

                if st.session_state.get(
                    "power_correct"
                ):

                    st.success(
                        "✅ Correct! More flips make the standard error "
                        "smaller, so a real bias stands out more clearly "
                        "from random variation."
                    )

                    st.latex(
                        r"n \uparrow \Rightarrow \text{Power} \uparrow"
                    )

                else:

                    st.error(
                        "Not quite. Think about what happens to the "
                        "standard error when n grows."
                    )

    navigation_buttons(
        show_continue=st.session_state.get(
            "power_correct",
            False
        )
    )


def render_final_challenge():
    """Render step 7 (final challenge) and its answer checks."""

    st.title(
        "Step 7: Final Challenge"
    )

    st.write(
        """
        You've now seen all the pieces of a hypothesis test.

        First, remember the reasoning process:
        """
    )

    st.info(
        """
        **Question → H₀/H₁ → Standard Error → Z-score →
        p-value → compare with α → decision → interpretation**
        """
    )

    st.write(
        """
        Now the scaffolding is removed.

        Solve the following new problem from start to finish.
        """
    )

    st.divider()

    st.header(
        "🚴 Delivery Time Investigation"
    )

    st.info(
        """
        A food delivery company claims that its average delivery time
        is **30 minutes**.

        Some customers believe that deliveries are actually taking
        **longer**.

        A random sample of **64 deliveries** has:

        **x̄ = 32 minutes**

        The population standard deviation is known to be:

        **σ = 8 minutes**

        Use:

        **α = 0.05**

        Determine whether the data provide sufficient evidence
        that the true average delivery time is greater than
        30 minutes.

        Explain your reasoning completely.
        """
    )

    st.subheader("Your workspace")

    # Streamlit drops a widget's state while its step is not shown, so keep
    # a copy to restore the answer after the student reviews a lesson step.
    if (
        "final_work" not in st.session_state
        and "final_work_saved" in st.session_state
    ):
        st.session_state.final_work = st.session_state.final_work_saved

    student_work = st.text_area(
        "Solve the problem from start to finish:",
        height=300,
        key="final_work",
        placeholder=(
            "Write your hypotheses, calculations, decision, "
            "and interpretation here..."
        )
    )
    st.session_state.final_work_saved = student_work

    # --------------------------------------------------------
    # Progressive hints
    # --------------------------------------------------------

    if "hint_number" not in st.session_state:
        st.session_state.hint_number = 0

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "💡 Need a hint",
            width="stretch"
        ):

            if st.session_state.hint_number < 4:
                st.session_state.hint_number += 1

    hints = [
        (
            "What claim should be treated as the "
            "default assumption?"
        ),
        (
            "Think about whether the alternative "
            "should use >, <, or ≠."
        ),
        (
            "Once you have the hypotheses, determine "
            "how far the observed sample mean is from H₀."
        ),
        (
            "For a sample mean with known population "
            "standard deviation: SE = σ / √n"
        )
    ]

    for i in range(
        st.session_state.hint_number
    ):
        st.info(
            f"💡 Hint {i + 1}: {hints[i]}"
        )

    # --------------------------------------------------------
    # Prototype "Check my work"
    # --------------------------------------------------------

    with col2:

        check_work = st.button(
            "✓ Check my work",
            type="primary",
            width="stretch"
        )

    ai_config = (
        get_ai_config()
        if FINAL_CHALLENGE_AI_FEEDBACK and openai is not None
        else None
    )

    if check_work:

        if not student_work.strip():

            st.warning(
                "Write some reasoning first so that your work can be checked."
            )

        elif ai_config:

            with st.spinner("Checking your work..."):
                try:
                    feedback = get_ai_feedback(student_work, *ai_config)
                except openai.AuthenticationError:
                    st.error("The AI key is invalid. Check RCP_API_KEY.")
                except openai.NotFoundError:
                    st.error(
                        "Model or endpoint not found. "
                        "Check RCP_MODEL and RCP_BASE_URL."
                    )
                except openai.RateLimitError:
                    st.error("Too many requests right now. Try again shortly.")
                except openai.APIConnectionError:
                    st.error(
                        "Could not reach the AI service. The EPFL RCP API "
                        "may require the EPFL network or VPN."
                    )
                except openai.APIStatusError as e:
                    st.error(f"AI feedback failed: {e}")
                except (ValueError, json.JSONDecodeError):
                    st.error(
                        "The AI feedback could not be read. "
                        "Please click “Check my work” again."
                    )
                else:
                    st.session_state.ai_feedback = feedback

        else:

            work = (
                student_work
                .lower()
                .replace(" ", "")
                .replace("μ", "mu")
                .replace("α", "alpha")
            )

            st.subheader("Feedback")

            # H0
            if (
                "h0:mu=30" in work
                or "h0=mu=30" in work
                or "h0:mean=30" in work
            ):
                st.success(
                    "✅ Your null hypothesis looks correct."
                )
            else:
                st.info(
                    "🔎 Check your null hypothesis. "
                    "What value represents the company's original claim?"
                )

            # H1
            if (
                "h1:mu>30" in work
                or "h1=mu>30" in work
                or "h1:mean>30" in work
            ):
                st.success(
                    "✅ Your alternative hypothesis has "
                    "the correct direction."
                )

            elif (
                "mu!=30" in work
                or "mu≠30" in work
                or "mu<30" in work
            ):
                st.info(
                    "🔎 Reconsider the direction of H₁. "
                    "The question specifically asks whether delivery "
                    "times are **greater than 30 minutes**."
                )

            else:
                st.info(
                    "🔎 Make sure you state an alternative hypothesis."
                )

            # SE
            if (
                "se=1" in work
                or "standarderror=1" in work
            ):
                st.success(
                    "✅ Your standard error appears to be correct."
                )
            else:
                st.info(
                    "🔎 Check the standard error using "
                    "**SE = σ / √n**."
                )

            # Z
            if (
                "z=2" in work
                or "z-score=2" in work
                or "zscore=2" in work
            ):
                st.success(
                    "✅ Your test statistic appears to be correct."
                )
            else:
                st.info(
                    "🔎 Once you have the standard error, calculate "
                    "how many standard errors 32 is above 30."
                )

            # p
            if (
                "0.0228" in work
                or "0.023" in work
                or "p=.0228" in work
            ):
                st.success(
                    "✅ Your p-value appears to be correct."
                )
            else:
                st.info(
                    "🔎 Check whether this is a left-tailed, "
                    "right-tailed, or two-sided test."
                )

            # decision
            if (
                "rejecth0" in work
                or "rejectthenull" in work
            ):
                st.success(
                    "✅ Your statistical decision appears correct."
                )
            else:
                st.info(
                    "🔎 Compare the p-value with α = 0.05."
                )

            # contextual conclusion
            if (
                "greaterthan30" in work
                or "longerthan30" in work
            ):
                st.success(
                    "✅ You appear to have interpreted the result "
                    "in the context of delivery times."
                )
            else:
                st.info(
                    "🔎 Finish with a conclusion about the "
                    "**true average delivery time**, not just "
                    "'reject H₀'."
                )

    if ai_config and st.session_state.get("ai_feedback"):
        render_ai_feedback(st.session_state.ai_feedback)

    # --------------------------------------------------------
    # Full solution
    # --------------------------------------------------------

    st.divider()

    with st.expander(
        "Show full solution"
    ):

        st.warning(
            "Try solving the problem yourself before opening the solution."
        )

        st.write("### 1. Hypotheses")

        st.latex(
            r"H_0: \mu = 30"
        )

        st.latex(
            r"H_1: \mu > 30"
        )

        st.write(
            "### 2. Standard Error"
        )

        st.latex(
            r"SE = \frac{\sigma}{\sqrt{n}}"
        )

        st.latex(
            r"SE = \frac{8}{\sqrt{64}} = 1"
        )

        st.write(
            "### 3. Test Statistic"
        )

        st.latex(
            r"Z = \frac{32-30}{1} = 2"
        )

        st.write(
            "### 4. p-value"
        )

        st.write(
            "Because \(H_1: \mu > 30\), this is a right-tailed test."
        )

        st.latex(
            r"p = P(Z \geq 2) \approx 0.0228"
        )

        st.write(
            "### 5. Decision"
        )

        st.latex(
            r"0.0228 < 0.05"
        )

        st.success(
            "Reject H₀"
        )

        st.write(
            "### 6. Real-world conclusion"
        )

        st.write(
            """
            There is statistically significant evidence that
            the true average delivery time is greater than
            30 minutes.
            """
        )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "← Previous",
            key="final_previous",
            width="stretch"
        ):
            st.session_state.step -= 1
            st.rerun()

    with col2:

        if st.button(
            "↺ Start Over",
            key="final_restart",
            width="stretch"
        ):
            restart_activity()
            st.rerun()

    with col3:

        if st.button(
            "Finish lesson →",
            key="final_finish",
            type="primary",
            width="stretch"
        ):
            st.session_state.step += 1
            st.rerun()


STATUS_ICONS = {
    "correct": "✅",
    "minor": "⚠️",
    "incorrect": "❌",
    "missing": "❌",
}


def go_to_review_step(step_number):
    """Send the student to a lesson step, remembering to come back."""
    st.session_state.review_return_step = st.session_state.step
    st.session_state.step = step_index(step_number)


def render_ai_feedback(feedback):
    """Show per-part AI feedback with links back to the lesson steps."""
    st.subheader("Feedback")

    parts = feedback["parts"]
    keys = [key for key, *_ in SOLUTION_PARTS]

    # Only give feedback up to the first part the student is stuck on;
    # later parts usually depend on it, so hints there would be premature.
    # "minor" issues are shown but do not stop the student.
    stop_index = next(
        (
            i for i, key in enumerate(keys)
            if parts[key]["status"] in ("incorrect", "missing")
        ),
        None,
    )
    # The student simply hasn't written this far yet: no hint, just a nudge.
    not_reached_yet = stop_index is not None and all(
        parts[key]["status"] == "missing" for key in keys[stop_index:]
    )

    for i, (key, label, lesson_step, lesson_label) in enumerate(
        SOLUTION_PARTS
    ):
        if stop_index is not None and i > stop_index:
            break
        part = parts[key]
        status = part["status"]

        if i == stop_index and not_reached_yet:
            st.info(
                f"➡️ **Next: {label}.** Everything up to here is checked. "
                f"Continue your solution with the {label.lower()}, then "
                "check your work again."
            )
            break

        with st.container(border=True):
            st.markdown(f"{STATUS_ICONS[status]} **{label}**")
            if part["feedback"]:
                st.markdown(part["feedback"])
            if status != "correct":
                st.button(
                    f"↩ Review Step {lesson_step}: {lesson_label}",
                    key=f"review_{key}",
                    on_click=go_to_review_step,
                    args=(lesson_step,),
                )

    if stop_index is not None and not not_reached_yet:
        remaining = len(keys) - stop_index - 1
        if remaining:
            st.caption(
                f"Fix this part first. The remaining {remaining} part"
                f"{'s' if remaining > 1 else ''} will be checked once "
                "it is right."
            )
    elif stop_index is None and feedback["overall"]:
        st.success(feedback["overall"])

    st.caption(
        "AI-generated feedback — it can make mistakes. Your answer is saved "
        "while you review a step; use “Back to Final Challenge” to return."
    )


def render_review_banner():
    """While reviewing an earlier step, offer a way back to the challenge."""
    return_step = st.session_state.get("review_return_step")
    if return_step is None:
        return
    if st.session_state.step == return_step:
        del st.session_state.review_return_step
        return

    with st.container(border=True):
        st.markdown(
            "📖 You're reviewing this step because of your final-challenge "
            "feedback. When you're ready, go back and fix your answer."
        )
        if st.button(
            "→ Back to Final Challenge",
            key="review_back",
            type="primary",
        ):
            st.session_state.step = return_step
            del st.session_state.review_return_step
            st.rerun()


# Add a new step function above, then append (label, function) here,
# and add its display number to STEP_NUMBERS below.
STEP_RENDERERS = (
    ("Consent", render_consent_step),
    ("Pre-test", render_pretest_step),
    ("Introduction", render_introduction),
    ("Data", render_data_step),
    ("Hypotheses", render_hypotheses_step),
    ("Z-score", render_test_statistic_step),
    ("p-value", render_pvalue_step),
    ("Decision", render_significance_step),
    ("Threshold", render_threshold_step),
    ("Errors & power", render_errors_step),
    ("Final challenge", render_final_challenge),
    ("Post-test", render_posttest_step),
    ("End", render_end_step),
)

# Number shown to learners for each entry of STEP_RENDERERS:
# "" = lesson introduction, None = study screen (no lesson progress bar).
STEP_NUMBERS = (None, None, "", "1", "2", "3", "4", "5", "5b", "6", "7",
                None, None)
assert len(STEP_NUMBERS) == len(STEP_RENDERERS)


def lesson_start_index():
    """Index of the lesson introduction in STEP_RENDERERS."""
    return STEP_NUMBERS.index("")


def step_index(number):
    """Index in STEP_RENDERERS of the lesson step shown as `number`."""
    return STEP_NUMBERS.index(str(number))


# Streamlit reruns this short entry point after every interaction.
st.set_page_config(
    page_title="Hypothesis Testing Activity",
    page_icon="🧪",
    layout="centered"
)


def render_dev_step_jumper():
    """Testing aid: jump to any step and switch version.

    Only with DEV_MODE = true in .streamlit/secrets.toml. Keep DEV_MODE out
    of the deployed app's secrets so participants never see this.
    """
    if not is_dev_mode():
        return

    with st.sidebar:
        st.caption("🛠 Developer mode")
        labels = [
            f"{STEP_NUMBERS[index]}. {label}" if STEP_NUMBERS[index]
            else label
            for index, (label, _) in enumerate(STEP_RENDERERS)
        ]
        target = st.selectbox(
            "Jump to step",
            range(len(STEP_RENDERERS)),
            index=st.session_state.step,
            format_func=lambda i: labels[i],
            key=f"dev_jump_{st.session_state.step}",
        )
        if target != st.session_state.step:
            st.session_state.step = target
            st.rerun()

        study = st.session_state.study
        condition = st.radio(
            "Version",
            CONDITIONS,
            index=CONDITIONS.index(study["condition"]),
            horizontal=True,
            key="dev_condition",
        )
        if condition != study["condition"]:
            study["condition"] = condition
            st.rerun()
        st.caption(
            f"Participant {study['participant_id']} · "
            f"forms {study['form_order']}"
        )


if not initialize_study_settings():
    st.title("Hypothesis Testing Activity")
    st.error(
        "This link is incomplete. Please open the exact link you received "
        "from the study team."
    )
    st.stop()

initialize_session_state()
render_dev_step_jumper()
render_progress()
render_review_banner()
_, render_current_step = STEP_RENDERERS[st.session_state.step]
st.session_state.learning_sections = []
render_current_step()
