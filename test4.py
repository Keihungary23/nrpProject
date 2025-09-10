import fitz  # PyMuPDF
from transformers import pipeline
from keybert import KeyBERT

# ================================
# 1. PDFからテキスト抽出
# ================================
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ================================
# 2. チャンク分割（オーバーラップ対応）
# ================================
def chunk_text(text, chunk_size=1500, overlap=200):
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunks.append(text[i:i + chunk_size])
        if i + chunk_size >= len(text):
            break
    return chunks

# ================================
# 3. 要約処理
# ================================
def summarize_chunks(chunks, summarizer):
    summaries = []
    for i, chunk in enumerate(chunks):
        try:
            summary = summarizer(
                chunk,
                max_length=200,
                min_length=40,
                do_sample=False
            )[0]["summary_text"]
            summaries.append(summary)
            print(f"✅ Chunk {i+1}/{len(chunks)} summarized.")
        except Exception as e:
            print(f"⚠️ Chunk {i+1} failed: {e}")
    return summaries

# ================================
# 4. 中間まとめ（章レベル）
# ================================
def summarize_sections(summaries, summarizer, chunk_size=1000):
    text = " ".join(summaries)
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    section_summaries = []
    for i, chunk in enumerate(chunks):
        summary = summarizer(
            chunk,
            max_length=300,
            min_length=60,
            do_sample=False
        )[0]["summary_text"]
        section_summaries.append(summary)
        print(f"📌 Section {i+1}/{len(chunks)} summarized.")
    return section_summaries

# ================================
# 5. キーワード抽出（n-gram & MMR対応）
# ================================
def extract_keywords(section_summaries, top_n=5):
    kw_model = KeyBERT()
    keywords_per_section = []
    for i, summary in enumerate(section_summaries):
        keywords = kw_model.extract_keywords(
            summary,
            top_n=top_n,
            use_mmr=True,                   # MMR有効化 → 多様性確保
            diversity=0.7,                  # 類似語を避ける度合い
            keyphrase_ngram_range=(1, 3)    # 1～3語のキーフレーズを許可
        )
        keywords_per_section.append(keywords)
        print(f"🔑 Keywords extracted for Section {i+1}")
    return keywords_per_section


# ================================
# 6. 最終まとめ（全体）
# ================================
def summarize_final(section_summaries, summarizer):
    final_input = " ".join(section_summaries)
    final_summary = summarizer(
        final_input,
        max_length=600,
        min_length=250,
        do_sample=False
    )[0]["summary_text"]
    return final_summary

# ================================
# 7. メイン処理
# ================================
if __name__ == "__main__":
    pdf_path = "1706.03762v7.pdf"   # Transformer 論文

    # テキスト抽出
    text = extract_text_from_pdf(pdf_path)
    print(f"📄 Extracted text length: {len(text)} characters")

    # チャンク分割
    chunks = chunk_text(text, chunk_size=1500)
    print(f"🔍 Split into {len(chunks)} chunks")

    # Hugging Face 要約モデル
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Step1: チャンク要約
    chunk_summaries = summarize_chunks(chunks, summarizer)

    # Step2: 中間まとめ（章レベル）
    section_summaries = summarize_sections(chunk_summaries, summarizer)

    # Step3: キーワード抽出
    keywords_per_section = extract_keywords(section_summaries, top_n=5)

    # Step4: 最終まとめ
    final_summary = summarize_final(section_summaries, summarizer)

    print("\n=== Section Summaries & Keywords ===\n")
    for i, (sec, keywords) in enumerate(zip(section_summaries, keywords_per_section), 1):
        print(f"[Section {i}] {sec}\n")
        print("   🔑 Keywords:", [kw for kw, score in keywords], "\n")

    print("\n=== Final Summary of the Paper ===\n")
    print(final_summary)
