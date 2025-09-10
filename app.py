import streamlit as st

st.write("✅ 起動テスト: ここまで動いた")

import streamlit as st
from transformers import pipeline
from nlp_utils import extract_text_from_pdf, chunk_text, summarize_chunks, summarize_sections, extract_keywords, summarize_final
from pdf_utils import highlight_pdf, display_pdf

st.title("📑 論文要約 & ハイライトビューア")

uploaded_file = st.file_uploader("PDFをアップロードしてください", type=["pdf"])

if uploaded_file:
    pdf_path = "uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # --- 要約処理 ---
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    chunk_summaries = summarize_chunks(chunks, summarizer)
    section_summaries = summarize_sections(chunk_summaries, summarizer)
    keywords_per_section = extract_keywords(section_summaries)
    final_summary = summarize_final(section_summaries, summarizer)

    st.subheader("📝 最終要約")
    st.write(final_summary)

    # --- PDFハイライト & 表示 ---
    all_keywords = [kw for section in keywords_per_section for kw, _ in section]
    highlighted_pdf = highlight_pdf(pdf_path, all_keywords)

    st.subheader("🔍 ハイライト付きPDFビューア")
    display_pdf(highlighted_pdf)

    with open(highlighted_pdf, "rb") as f:
        st.download_button("⬇ ダウンロード", f, file_name="highlighted.pdf")
