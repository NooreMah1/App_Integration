import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
_api_key = os.getenv("GROQ_API_KEY")
_client = Groq(api_key=_api_key) if _api_key else None


def build_prompt(question, model_answer, student_answer):
    return f"""
You are a strict examiner grading a written exam answer. Your job is to catch
shallow/incomplete answers, not to be generous. Most student answers deserve
a MIDDLE or LOW score — a high score should be rare and only for genuinely
complete, accurate answers.
 
CRITICAL RULE — READ CAREFULLY:
Sharing a few words or phrases with the MODEL ANSWER does NOT mean the student
understood the concept. Many weak answers reuse a term from the question or
model answer without explaining it correctly. You must actively check for this
and NOT reward it. Award marks only for concepts the student has correctly
EXPLAINED in their own words — not for vocabulary overlap.
 
Grade using this breakdown (mentally check each one, then decide the score):
1. Does the student correctly define/explain the core concept? (most important)
2. Are the key points from the model answer actually present and accurate?
3. Is anything factually wrong or contradictory?
4. Is the answer missing major parts of what was asked?
 
Use this scoring band as a guide (do not just default to the middle):
- 9-10: Fully correct, complete, matches all key points in the model answer
- 7-8: Mostly correct, minor omissions or slightly incomplete detail
- 5-6: Partially correct — gets the general idea but misses key points or has some inaccuracy
- 3-4: Weak — only a superficial/keyword-level resemblance to the correct answer, most of the substance is missing or wrong
- 0-2: Incorrect, irrelevant, or blank
 
Do NOT penalize different wording if the meaning is fully and correctly conveyed.
Do NOT give credit for correct-sounding phrases, buzzwords, or terms copied from
the question/model answer that are not actually explained correctly.

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

