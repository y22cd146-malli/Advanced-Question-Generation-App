# Advanced Question Generation App

This is a web-based app built using Gradio that:
- Uploads `.txt` lesson files
- Automatically generates questions using T5
- Answers them using a QA model (DistilBERT)
- Allows user to download the Q&A as a `.txt` file

## Requirements

```bash
pip install transformers gradio torch spacy
python -m spacy download en_core_web_sm
```

## Run the App

```bash
python app.py
```

Then open the link shown in your terminal to access the web interface.
