import spacy
import re
import gradio as gr
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import torch

# Load models (use GPU if available)
device = 0 if torch.cuda.is_available() else -1
nlp = spacy.load("en_core_web_sm")
t5_model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-qg-hl")
t5_tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-base-qg-hl")
question_answerer = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=device)

# Highlight the first occurrence of the answer
def highlight_answer(text, answer):
    return re.sub(re.escape(answer), f"<hl> {answer} <hl>", text, count=1)

# Generate questions from entities and noun chunks
def generate_questions(text, max_questions=5):
    doc = nlp(text)
    questions = []
    # Filter: Only useful named entities and meaningful noun chunks
    candidates = list(set([ent.text for ent in doc.ents if ent.label_ in ("PERSON", "ORG", "GPE", "DATE", "EVENT")] +
                          [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 4]))

    for answer in candidates:
        if len(questions) >= max_questions or answer not in text:
            continue
        input_text = "generate question: " + highlight_answer(text, answer) + " </s>"
        input_ids = t5_tokenizer.encode(input_text, return_tensors="pt")
        outputs = t5_model.generate(input_ids=input_ids, max_length=64, num_beams=4, early_stopping=True)
        question = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        if question and question not in questions:
            questions.append((question, answer))
    return questions

# Answer questions using QA pipeline
def answer_questions(text, questions):
    return [(q, question_answerer(question=q, context=text)["answer"]) for q, _ in questions]

# Main Gradio function
def process_file(file, max_questions):
    text = file.read().decode("utf-8")
    qas = generate_questions(text, max_questions=max_questions)
    answered = answer_questions(text, qas)
    output = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in answered])
    return output, output.encode("utf-8")

# Gradio UI
iface = gr.Interface(
    fn=process_file,
    inputs=[
        gr.File(label="Upload a .txt lesson file"),
        gr.Slider(1, 15, value=5, step=1, label="Number of Questions")
    ],
    outputs=[
        gr.Textbox(label="Generated Questions and Answers", lines=15),
        gr.File(label="Download Q&A (.txt)", file_types=[".txt"])
    ],
    title="Advanced Question Generator",
    description="Upload a .txt file and get generated questions and answers using NLP and transformers."
)

if __name__ == "__main__":
    iface.launch()
