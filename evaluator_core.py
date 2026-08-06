"""
evaluator_core.py
------------------
Aapke original answer_evaluator.py ka SAME logic hai — bas ipywidgets
(Jupyter UI) hata diya hai, taake ye function seedha PyQt app se call
ho sake. Groq API call, prompt, parsing — sab hooba-hoo wahi hai.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
_api_key = os.getenv("GROQ_API_KEY")
_client = Groq(api_key=_api_key) if _api_key else None


def build_prompt(question, model_answer, student_answer):
    return f"""
You are a strict but fair examiner evaluating a student's written answer.

Judge based on: factual accuracy, completeness, and understanding (not just keyword matching).
Do NOT penalize different wording if the meaning is correct.
Do NOT give credit for correct-sounding phrases that show no real understanding.

QUESTION:
{question}

MODEL ANSWER:
{model_answer}

STUDENT ANSWER:
{student_answer}

Return your evaluation in EXACTLY this format, nothing else:

Score: X/10
Remarks:
- (short bullet point 1)
- (short bullet point 2)
- (short bullet point 3, only if applicable)
- (short bullet point 4, only if applicable)

Write each remark in direct instructor style — as if a teacher is marking the answer directly.
Do NOT start remarks with phrases like "The student..." or "The student's answer...".
Keep each remark short and direct, like a real margin note on a graded paper.
"""


def get_evaluation(prompt):
    if _client is None:
        return None
    try:
        response = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Groq API call failed: {e}")
        return None


def parse_response(response_text):
    if not response_text:
        return None, []
    lines = response_text.strip().split("\n")
    score, remarks = None, []
    for line in lines:
        line = line.strip()
        if line.lower().startswith("score:"):
            try:
                score = int(line.split(":")[1].strip().split("/")[0].strip())
            except (ValueError, IndexError):
                score = None
        elif line.startswith("-"):
            remarks.append(line.lstrip("-").strip())
    return score, remarks


def evaluate_answer(question, model_answer, student_answer):
    """Master function — GUI yahi call karega.
    Returns: {"score": int|None, "remarks": [...], "error": str|None}
    """
    prompt = build_prompt(question, model_answer, student_answer)
    raw_response = get_evaluation(prompt)

    if raw_response is None:
        return {"score": None, "remarks": [],
                "error": "Could not reach the AI service. Please check your API key or internet connection."}

    score, remarks = parse_response(raw_response)

    if score is None:
        return {"score": None, "remarks": [],
                "error":  "Could not understand the AI response. Please try again."}

    return {"score": score, "remarks": remarks, "error": None}

