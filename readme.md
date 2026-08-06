# AI Answer Marking System

An AI-powered desktop application that automates the marking of handwritten student answer sheets. It detects answer boxes in a PDF, extracts handwritten text using an AI vision model (OCR), evaluates each answer against a model answer using an LLM, and stamps the score and remarks directly back onto the original PDF.

## How It Works

```
PDF Upload → Box Detection (OpenCV) → OCR (OpenRouter vision model)
   → Question/Model Answer input (GUI) → Evaluation (Groq LLaMA-3.3-70B)
      → Score + Remarks stamped back onto PDF (overlay technique)
```

1. **`pdf_extractor.py`** — Renders each PDF page to an image, detects bordered answer boxes using OpenCV (edge detection + contours), and crops each box into a separate image.
2. **`ocr_engine.py`** — Sends each cropped image to a free AI vision model via OpenRouter to extract the handwritten text, with automatic fallback across multiple models.
3. **`evaluator_core.py`** — Sends the question, model answer, and extracted student answer to Groq's LLaMA-3.3-70B model, which returns a score (0–10) and instructor-style remarks.
4. **`autofiller.py`** — Re-detects the same boxes in the original PDF and stamps the score/remarks onto the correct box using a transparent overlay layer, merged with the original PDF.
5. **`marking_app.py`** — PyQt5 desktop GUI that orchestrates the full pipeline: upload → extract & OCR → review/evaluate each answer → generate the final marked PDF.

## Tech Stack

- **GUI**: PyQt5
- **Computer Vision**: OpenCV, PyMuPDF (fitz)
- **OCR**: OpenRouter free vision models (Gemma, Nemotron)
- **Answer Evaluation**: Groq API (LLaMA-3.3-70B)
- **PDF Generation**: pypdf, ReportLab

## Setup

```bash
git clone <this-repo-url>
cd ai-answer-marking-system
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root with your API keys:

```
GROQ_API_KEY=your_groq_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

- Get a free Groq key at [console.groq.com](https://console.groq.com)
- Get a free OpenRouter key at [openrouter.ai](https://openrouter.ai)

## Usage

```bash
python marking_app.py
```

1. **Upload PDF** — select the scanned answer sheet
2. **Step 1: Extract + OCR** — detects answer boxes and runs OCR on each
3. Select an extracted answer → enter the **Question** and **Model Answer** → click **Evaluate This Answer**
4. Repeat for each answer, then click **Step 3: Generate Marked PDF** to produce the final scored PDF

## Key Design Notes

- Box detection order (top-to-bottom, left-to-right) determines the answer numbering — this order is identical across the extraction and stamping stages, so no filename parsing is needed to map results back onto the original PDF.
- All PDF rendering uses a consistent DPI (200) across modules, since coordinate mapping between PDF space and image space depends on it.

## Project Structure

```
├── pdf_extractor.py     # Box detection + image cropping
├── ocr_engine.py         # Vision-model OCR with fallback
├── evaluator_core.py     # LLM-based answer evaluation
├── autofiller.py          # PDF overlay stamping
├── marking_app.py         # PyQt5 GUI (entry point)
├── requirements.txt
└── .env                   # API keys (not committed)
```