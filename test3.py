import fitz  # PyMuPDF
from transformers import pipeline

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
# 2. チャンク分割
# ================================
def chunk_text(text, chunk_size=1500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# ================================
# 3. 要約処理
# ================================
def summarize_chunks(chunks, summarizer):
    summaries = []
    for i, chunk in enumerate(chunks):
        try:
            summary = summarizer(
                chunk,
                max_length=150,   # 出力の長さを調整
                min_length=40,
                do_sample=False
            )[0]["summary_text"]
            summaries.append(summary)
            print(f"✅ Chunk {i+1}/{len(chunks)} summarized.")
        except Exception as e:
            print(f"⚠️ Chunk {i+1} failed: {e}")
    return summaries

def summarize_final(summaries, summarizer, chunk_size=800):
    # summaries の結合文をさらに小分けして処理
    text = " ".join(summaries)
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    mids = []
    for ch in chunks:
        mids.append(
            summarizer(ch, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
        )
    # 中間まとめを最終結合
    final_input = " ".join(mids)
    final_summary = summarizer(final_input, max_length=250, min_length=80, do_sample=False)[0]["summary_text"]
    return final_summary


# ================================
# 4. メイン処理
# ================================
if __name__ == "__main__":
    pdf_path = "1706.03762v7.pdf"   # すでにあるPDFを指定

    # テキスト抽出
    text = extract_text_from_pdf(pdf_path)
    print(f"📄 Extracted text length: {len(text)} characters")

    # チャンク分割
    chunks = chunk_text(text, chunk_size=1500)
    print(f"🔍 Split into {len(chunks)} chunks")

    # Hugging Face 要約モデル準備
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # 各チャンク要約
    chunk_summaries = summarize_chunks(chunks, summarizer)

    # 最終まとめ要約
    final_summary = summarize_final(chunk_summaries, summarizer)

    print("\n=== Final Summary of the Paper ===\n")
    print(final_summary)
