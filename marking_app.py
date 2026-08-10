
from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSplitter,
    QTextEdit, QVBoxLayout, QWidget,
)

import pdf_extractor
import ocr_engine
import evaluator_core
import autofiller


IMAGES_DIR_NAME = pdf_extractor.DEFAULT_IMAGES_DIR_NAME
DEFAULT_DPI = pdf_extractor.DEFAULT_DPI
DEFAULT_DELAY_BETWEEN_IMAGES = ocr_engine.DEFAULT_DELAY_BETWEEN_IMAGES


class MarkingWorker(QThread):
    log_message = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, pdf_path, images_dir, api_key,
                 delay_between_images=DEFAULT_DELAY_BETWEEN_IMAGES):
        super().__init__()
        self.pdf_path = pdf_path
        self.images_dir = images_dir
        self.api_key = api_key
        self.delay_between_images = delay_between_images

    def run(self):
        try:
            if not self.api_key:
                raise RuntimeError("No OpenRouter API key provided.")

            self.log_message.emit("Step 1/2: Extracting answer boxes from PDF...")
            image_paths = pdf_extractor.extract_answer_images(
                self.pdf_path, output_dir=self.images_dir,
                dpi=DEFAULT_DPI, log=self.log_message.emit,
            )
            if not image_paths:
                raise RuntimeError("No answer boxes were detected in the PDF.")

            self.log_message.emit(f"Extracted {len(image_paths)} image(s).")
            self.log_message.emit("Step 2/2: Running OCR on extracted images...")
            results = ocr_engine.extract_text_from_images(
                image_paths, api_key=self.api_key, log=self.log_message.emit,
                progress=self.progress.emit, delay_between_images=self.delay_between_images,
            )
            self.finished_ok.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Automated Marking")
        self.resize(1100, 650)

        self.pdf_path: str | None = None
        self.results: list = []          # OCR results, order = answer_no - 1
        self.evaluations: dict = {}       # {answer_no: {"score":.., "remarks":[...]}}
        self.worker: MarkingWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # --- Top row: PDF selection ---
        top_row = QHBoxLayout()
        self.upload_btn = QPushButton("Upload PDF")
        self.upload_btn.clicked.connect(self.on_upload_pdf)
        self.pdf_label = QLabel("No PDF selected")
        top_row.addWidget(self.upload_btn)
        top_row.addWidget(self.pdf_label, stretch=1)
        root_layout.addLayout(top_row)

        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("OpenRouter API Key:"))
        self.api_key_edit = QLineEdit(os.environ.get("OPENROUTER_API_KEY", ""))
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        key_row.addWidget(self.api_key_edit, stretch=1)
        root_layout.addLayout(key_row)

        action_row = QHBoxLayout()
        self.mark_btn = QPushButton("Step 1: Extract + OCR")
        self.mark_btn.setEnabled(False)
        self.mark_btn.clicked.connect(self.on_do_marking)
        self.progress_bar = QProgressBar()
        action_row.addWidget(self.mark_btn)
        action_row.addWidget(self.progress_bar, stretch=1)
        root_layout.addLayout(action_row)

        splitter = QSplitter(Qt.Horizontal)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        splitter.addWidget(self._with_label("Log", self.log_view))

        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        results_layout.addWidget(QLabel("Extracted Answers"))
        self.results_list = QListWidget()
        self.results_list.currentItemChanged.connect(self.on_result_selected)
        results_layout.addWidget(self.results_list, stretch=1)
        splitter.addWidget(results_panel)

        # --- NAYA PANEL: extracted text + question + model answer + evaluate ---
        eval_panel = QWidget()
        eval_layout = QVBoxLayout(eval_panel)

        eval_layout.addWidget(QLabel("Student Answer (OCR — editable)"))
        self.student_answer_box = QTextEdit()
        self.student_answer_box.setMaximumHeight(100)
        eval_layout.addWidget(self.student_answer_box)

        eval_layout.addWidget(QLabel("Question"))
        self.question_box = QTextEdit()
        self.question_box.setMaximumHeight(60)
        eval_layout.addWidget(self.question_box)

        eval_layout.addWidget(QLabel("Model Answer"))
        self.model_answer_box = QTextEdit()
        self.model_answer_box.setMaximumHeight(80)
        eval_layout.addWidget(self.model_answer_box)

        self.evaluate_btn = QPushButton("Evaluate This Answer")
        self.evaluate_btn.setEnabled(False)
        self.evaluate_btn.clicked.connect(self.on_evaluate_clicked)
        eval_layout.addWidget(self.evaluate_btn)

        self.eval_result_label = QLabel("")
        self.eval_result_label.setWordWrap(True)
        eval_layout.addWidget(self.eval_result_label)

        eval_layout.addStretch(1)
        splitter.addWidget(eval_panel)

        splitter.setSizes([280, 250, 400])
        root_layout.addWidget(splitter, stretch=1)

        bottom_row = QHBoxLayout()
        self.generate_pdf_btn = QPushButton("Step 3: Generate Marked PDF")
        self.generate_pdf_btn.setEnabled(False)
        self.generate_pdf_btn.clicked.connect(self.on_generate_pdf)
        bottom_row.addStretch(1)
        bottom_row.addWidget(self.generate_pdf_btn)
        root_layout.addLayout(bottom_row)

    def _with_label(self, title, widget):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(QLabel(title))
        layout.addWidget(widget, stretch=1)
        return container

    # ---------------- PDF Upload ----------------
    def on_upload_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select PDF", "", "PDF Files (*.pdf)")
        if path:
            self.pdf_path = path
            self.pdf_label.setText(path)
            self.mark_btn.setEnabled(True)

    # ---------------- Step 1+2: Extract + OCR ----------------
    def on_do_marking(self):
        if not self.pdf_path:
            QMessageBox.warning(self, "No PDF", "Please upload a PDF first.")
            return
        api_key = self.api_key_edit.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Missing API Key", "Enter OpenRouter API key.")
            return

        self.log_view.clear()
        self.results_list.clear()
        self.results = []
        self.evaluations = {}
        self.progress_bar.setValue(0)
        self.mark_btn.setEnabled(False)

        images_dir = str(Path.cwd() / IMAGES_DIR_NAME)
        self.worker = MarkingWorker(self.pdf_path, images_dir, api_key)
        self.worker.log_message.connect(self.append_log)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished_ok.connect(self.on_pipeline_finished)
        self.worker.failed.connect(self.on_pipeline_failed)
        self.worker.start()

    def append_log(self, message):
        self.log_view.appendPlainText(message)

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def on_pipeline_finished(self, results):
        self.results = results
        self.mark_btn.setEnabled(True)

        for i, r in enumerate(results, start=1):
            status_icon = "OK" if r["status"] == "success" else "FAIL"
            item = QListWidgetItem(f"[{status_icon}] Answer {i} — {r['image']}")
            self.results_list.addItem(item)

        self.append_log(f"\nReady. {len(results)} answer(s) extracted — select one to evaluate.")

    def on_pipeline_failed(self, message):
        self.mark_btn.setEnabled(True)
        self.append_log(f"\nERROR: {message}")
        QMessageBox.critical(self, "Failed", message)

    # ---------------- Step 2b: Select + Evaluate ----------------
    def on_result_selected(self, current, _previous):
        if current is None:
            return
        index = self.results_list.row(current)          # 0-based
        result = self.results[index]

        if result["status"] == "success":
            self.student_answer_box.setPlainText(result["extracted_text"])
            self.evaluate_btn.setEnabled(True)
        else:
            self.student_answer_box.setPlainText(f"[OCR Error] {result.get('error')}")
            self.evaluate_btn.setEnabled(False)

        answer_no = index + 1
        if answer_no in self.evaluations:
            ev = self.evaluations[answer_no]
            self.eval_result_label.setText(
                f"Score: {ev['score']}/10\nRemarks:\n- " + "\n- ".join(ev["remarks"])
            )
        else:
            self.eval_result_label.setText("(not evaluated yet)")

    def on_evaluate_clicked(self):
        current = self.results_list.currentItem()
        if current is None:
            return
        index = self.results_list.row(current)
        answer_no = index + 1

        question = self.question_box.toPlainText().strip()
        model_answer = self.model_answer_box.toPlainText().strip()
        student_answer = self.student_answer_box.toPlainText().strip()

        if not question or not model_answer or not student_answer:
            QMessageBox.warning(self, "Missing Info", "Fill all required fields.")
            return

        self.eval_result_label.setText("Evaluating...")
        QApplication.processEvents()

        result = evaluator_core.evaluate_answer(question, model_answer, student_answer)

        if result["error"]:
            self.eval_result_label.setText(f"Error: {result['error']}")
            return

        self.evaluations[answer_no] = {"score": result["score"], "remarks": result["remarks"]}
        self.eval_result_label.setText(
            f"Score: {result['score']}/10\nRemarks:\n- " + "\n- ".join(result["remarks"])
        )

        # Update list item to show it's evaluated
        current.setText(f"[Evaluated: {result['score']}/10] Answer {answer_no} — {self.results[index]['image']}")

        # Enable final step once at least one answer is evaluated
        self.generate_pdf_btn.setEnabled(True)

    # ---------------- Step 3: Generate Marked PDF ----------------
    def on_generate_pdf(self):
        if not self.evaluations:
            QMessageBox.warning(self, "Nothing evaluated", "Evaluate atleast one answer.")
            return

        scoring_list = [
            {"answer_no": answer_no, "score": ev["score"], "remarks": ev["remarks"]}
            for answer_no, ev in self.evaluations.items()
        ]

        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Marked PDF", "marked_output.pdf", "PDF Files (*.pdf)"
        )
        if not output_path:
            return

        self.append_log("\nGenerating marked PDF...")
        try:
            autofiller.generate_marked_pdf(
                self.pdf_path, scoring_list, output_path, log=self.append_log
            )
            QMessageBox.information(self, "Done", f"Marked PDF saved to:\n{output_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Failed", str(exc))


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()