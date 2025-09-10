from transformers import pipeline

# ================================
# 1. モデル準備
# ================================
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ================================
# 2. チャンク分割関数
# ================================
def chunk_text(text, chunk_size=1500):
    """テキストを chunk_size ごとに分割"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ================================
# 3. 各チャンクの要約
# ================================
def summarize_chunks(chunks):
    summaries = []
    for i, chunk in enumerate(chunks):
        try:
            summary = summarizer(
                chunk,
                max_length=150,   # 出力の長さ（調整可）
                min_length=40,
                do_sample=False
            )[0]["summary_text"]
            summaries.append(summary)
            print(f"✅ Chunk {i+1} summarized.")
        except Exception as e:
            print(f"⚠️ Chunk {i+1} failed: {e}")
    return summaries

# ================================
# 4. 最終要約
# ================================
def summarize_final(summaries):
    final_input = " ".join(summaries)
    final_summary = summarizer(
        final_input,
        max_length=300,   # 全体まとめの長さ
        min_length=100,
        do_sample=False
    )[0]["summary_text"]
    return final_summary


# ================================
# 5. 実行例
# ================================
if __name__ == "__main__":
    # サンプルテキスト（本番ではPDF抽出した全文をここに渡す）
    text = """Deep learning has revolutionized natural language processing...
              (ここに論文テキスト全文を入れる)"""

    # チャンク分割
    chunks = chunk_text(text, chunk_size=1500)

    # チャンクごとに要約
    chunk_summaries = summarize_chunks(chunks)

    # 最終まとめ要約
    final_summary = summarize_final(chunk_summaries)

    print("\n=== Final Summary ===\n")
    print(final_summary)
