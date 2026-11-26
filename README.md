# AI Document Intelligence

AI-assisted educational NLP prototype for classifying synthetic business documents, extracting structured fields, and searching text.

## Run locally
Python 3.10+ recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python engine.py
python -m unittest -v
```

## Architecture
Synthetic text → TF-IDF features → logistic regression classification; regular expressions extract references, amounts and date strings; cosine similarity ranks keyword-based search results.

## Evaluation
160 synthetic documents, 120 training and 40 holdout records. The random split shares repetitive templates across train and test, so scores demonstrate functionality rather than generalization. No real documents or personal data are included. This is lexical search, not embedding-based semantic search. No PDF parsing or OCR is implemented; input is plain text. Extracted dates are strings, not validated calendar dates. Seven software tests cover core functionality.

## Practice tasks
Add new unseen templates, split by template family, measure per-class precision and recall, and improve extraction validation. Review and execute the code yourself before representing it as a demonstrated skill.
