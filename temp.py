import fitz  # PyMuPDF

# PDFを開く
doc = fitz.open("thesis_documentation.pdf")

# 全ページからテキストを抽出
text = ""
for page in doc:
    text += page.get_text()

# 抽出結果を表示
print(text[:1000])  # 最初の1000文字だけ表示
