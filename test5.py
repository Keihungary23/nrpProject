# optimize_summarize.py
import os
import time
import hashlib
import json
import fitz  # PyMuPDF
from transformers import pipeline, AutoTokenizer
from keybert import KeyBERT

# ------------------------
# PDF -> text
# ------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ------------------------
# tokenベースのチャンク化
# ------------------------
def chunk_text_by_tokens(text, tokenizer, token_chunk_size=800, token_overlap=100):
    # tokenizer.encode returns list of token ids
    token_ids = tokenizer.encode(text)
    chunks = []
    step = token_chunk_size - token_overlap
    for i in range(0, len(token_ids), step):
        chunk_ids = token_ids[i:i+token_chunk_size]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
        if i + token_chunk_size >= len(token_ids):
            break
    return chunks

# ------------------------
# キャッシュユーティリティ
# ------------------------
def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def load_cache(fp):
    if os.path.exists(fp):
        with open(fp, "rb") as f:
            return json.load(f)
    return None

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ------------------------
# 要約（チャンク）
# ------------------------
def summarize_chunks(chunks, summarizer, max_len=100, min_len=20):
    summaries = []
    t0 = time.time()
    for i, chunk in enumerate(chunks, 1):
        t1 = time.time()
        summary = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
        summaries.append(summary)
        dt = time.time() - t1
        print(f"✅ Chunk {i}/{len(chunks)} summarized (time {dt:.2f}s).")
    print("Total chunk summarization time:", time.time()-t0)
    return summaries

# ------------------------
# 中間まとめ（section） - tokenベースで分割することも可
# ------------------------
def summarize_sections(summaries, summarizer, token_chunk_size=1000, token_overlap=100, tokenizer=None, max_len=200, min_len=40):
    text = " ".join(summaries)
    if tokenizer:
        sec_chunks = chunk_text_by_tokens(text, tokenizer, token_chunk_size, token_overlap)
    else:
        # fallback: character chunks
        sec_chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
    section_summaries = []
    for i, sec in enumerate(sec_chunks, 1):
        s = summarizer(sec, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
        section_summaries.append(s)
        print(f"📌 Section {i}/{len(sec_chunks)} summarized.")
    return section_summaries

# ------------------------
# キーワード抽出（KeyBERT, ngram+MMR）
# ------------------------
def extract_keywords(section_summaries, top_n=6):
    kw_model = KeyBERT()  # note: first load downloads model
    keywords_per_section = []
    for i, summary in enumerate(section_summaries, 1):
        kws = kw_model.extract_keywords(
            summary,
            top_n=top_n,
            use_mmr=True,
            diversity=0.7,
            keyphrase_ngram_range=(1, 3)
        )
        keywords_per_section.append(kws)
        print(f"🔑 Keywords extracted for Section {i}")
    return keywords_per_section

# ------------------------
# 最終まとめ
# ------------------------
def summarize_final(section_summaries, summarizer, max_len=400, min_len=150):
    final_input = " ".join(section_summaries)
    final_summary = summarizer(final_input, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
    return final_summary

# ------------------------
# Main: 最適化フロー
# ------------------------
if __name__ == "__main__":
    pdf_path = "1706.03762v7.pdf"
    text = extract_text_from_pdf(pdf_path)
    print(f"📄 Extracted text length: {len(text)} characters")

    fp = fingerprint(text)
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    chunk_cache = os.path.join(cache_dir, f"{fp}_chunks.json")
    chunk_summary_cache = os.path.join(cache_dir, f"{fp}_chunk_summaries.json")
    section_summary_cache = os.path.join(cache_dir, f"{fp}_section_summaries.json")
    final_cache = os.path.join(cache_dir, f"{fp}_final.json")

    # tokenizer & summarizer (軽量モデル)
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    # チャンク化（tokenベース）
    if os.path.exists(chunk_cache):
        print("⚡ load chunks from cache")
        chunks = load_cache(chunk_cache)
    else:
        chunks = chunk_text_by_tokens(text, tokenizer, token_chunk_size=800, token_overlap=100)
        save_json(chunks, chunk_cache)
        print(f"🔍 Split into {len(chunks)} chunks (token-based)")

    # チャンク要約（キャッシュ有効）
    if os.path.exists(chunk_summary_cache):
        print("⚡ load chunk summaries from cache")
        chunk_summaries = load_cache(chunk_summary_cache)
    else:
        start = time.time()
        chunk_summaries = summarize_chunks(chunks, summarizer, max_len=100, min_len=20)
        save_json(chunk_summaries, chunk_summary_cache)
        print("Chunk summaries time:", time.time()-start)

    # Section summaries
    if os.path.exists(section_summary_cache):
        section_summaries = load_cache(section_summary_cache)
    else:
        start = time.time()
        section_summaries = summarize_sections(chunk_summaries, summarizer, token_chunk_size=1200, token_overlap=200, tokenizer=tokenizer, max_len=200, min_len=60)
        save_json(section_summaries, section_summary_cache)
        print("Section summaries time:", time.time()-start)

    # keywords
    keywords_file = os.path.join(cache_dir, f"{fp}_keywords.json")
    if os.path.exists(keywords_file):
        keywords_per_section = load_cache(keywords_file)
    else:
        keywords_per_section = extract_keywords(section_summaries, top_n=6)
        # serialize as list of lists (keyword, score)
        save_json([[ (k,s) for k,s in sec ] for sec in keywords_per_section], keywords_file)

    # final
    if os.path.exists(final_cache):
        final_summary = load_cache(final_cache)
    else:
        final_summary = summarize_final(section_summaries, summarizer, max_len=400, min_len=120)
        save_json(final_summary, final_cache)

    print("\n=== Final Summary ===\n")
    print(final_summary)
