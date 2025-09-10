# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

app = FastAPI()

# モデル初期化
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
kw_model = KeyBERT(model=SentenceTransformer('all-MiniLM-L6-v2'))

class InputText(BaseModel):
    text: str

@app.post("/summarize")
def summarize(input: InputText):
    # 要約
    summary = summarizer(input.text[:1000], max_length=120, min_length=30, do_sample=False)[0]['summary_text']
    # キーワード
    keywords = [kw[0] for kw in kw_model.extract_keywords(input.text, top_n=10)]
    return {"summary": summary, "keywords": keywords}
