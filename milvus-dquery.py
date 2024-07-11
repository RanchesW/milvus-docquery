import tkinter as tk
from tkinter import filedialog, scrolledtext
from pdf2image import convert_from_path
import pytesseract
from transformers import BertTokenizer, BertModel
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# Initialize Milvus and BERT
def init_milvus_and_bert():
    connections.connect(host='localhost', port='19530')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

tokenizer, model = init_milvus_and_bert()

# Milvus schema and collection configuration
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields, description="PDF Text Search Collection")
collection_name = "pdf_text_search"
collection = Collection(name=collection_name, schema=schema)

def pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)

def images_to_text(images):
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang='eng')
    return text

def text_to_vector(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy().flatten().tolist()

def open_pdf():
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if file_path:
        images = pdf_to_images(file_path)
        text = images_to_text(images)
        return text
    return ""

def handle_query():
    query_text = entry_text.get()
    query_vector = text_to_vector(query_text)
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.search(data=[query_vector], anns_field="text_vector", param=search_params, limit=5, expr=None)
    output = ""
    for hits in results:
        for hit in hits:
            output += f"Hit ID: {hit.id}, Distance: {hit.distance}\n"
    output_text.config(state=tk.NORMAL)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, output)
    output_text.config(state=tk.DISABLED)

root = tk.Tk()
root.title("PDF Search System")

entry_text = tk.StringVar()
tk.Label(root, text="Введите текст для поиска:").pack()
query_var = tk.Entry(root, textvariable=entry_text, width=70)
query_var.pack()

open_button = tk.Button(root, text="Открыть PDF", command=lambda: entry_text.set(open_pdf()))
open_button.pack()

search_button = tk.Button(root, text="Поиск", command=handle_query)
search_button.pack()

output_text = scrolledtext.ScrolledText(root, width=100, height=20)
output_text.pack()
output_text.config(state=tk.DISABLED)

root.mainloop()
