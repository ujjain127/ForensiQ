import os
import pandas as pd
from docx import Document
import fitz  # PyMuPDF
import json

# 🔧 Path to your raw data folder
RAW_DATA_FOLDER = 'data/raw'

# 📁 Output folder
OUTPUT_FOLDER = 'data/processed'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def parse_json(file_path):
    if file_path.endswith('.jsonl'):
        # Handle JSONL (JSON Lines) format
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"⚠️ JSON error on line {line_num}: {e}")
                            continue
            print(f"✅ Successfully parsed {len(data)} JSON objects from JSONL")
            return pd.json_normalize(data)
        except Exception as e:
            print(f"❌ Error reading JSONL file: {e}")
            return pd.DataFrame()
    else:
        # Handle regular JSON format
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return pd.json_normalize(data)
        except Exception as e:
            print(f"❌ Error reading JSON file: {e}")
            return pd.DataFrame()

def parse_csv(file_path):
    return pd.read_csv(file_path)

def parse_txt(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def parse_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def parse_docx(file_path):
    doc = Document(file_path)
    return '\n'.join([p.text for p in doc.paragraphs])

def save_text_output(content, filename):
    with open(os.path.join(OUTPUT_FOLDER, filename), 'w', encoding='utf-8') as f:
        f.write(content)

def save_dataframe_output(df, filename):
    df.to_csv(os.path.join(OUTPUT_FOLDER, filename), index=False)

def parse_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    print(f"Processing: {file_path} (ext: {ext})")
    try:
        if ext == '.json' or ext == '.jsonl':
            df = parse_json(file_path)
            output_name = os.path.basename(file_path).replace(ext, '_parsed.csv')
            save_dataframe_output(df, output_name)
            print(f"✅ Saved: {output_name}")
        elif ext == '.csv':
            df = parse_csv(file_path)
            output_name = os.path.basename(file_path).replace('.csv', '_parsed.csv')
            save_dataframe_output(df, output_name)
            print(f"✅ Saved: {output_name}")
        elif ext == '.txt' or ext == '.log':
            text = parse_txt(file_path)
            output_name = os.path.basename(file_path).replace(ext, '_parsed.txt')
            save_text_output(text, output_name)
            print(f"✅ Saved: {output_name}")
        elif ext == '.pdf':
            text = parse_pdf(file_path)
            output_name = os.path.basename(file_path).replace('.pdf', '_parsed.txt')
            save_text_output(text, output_name)
            print(f"✅ Saved: {output_name}")
        elif ext == '.doc' or ext == '.docx':
            text = parse_docx(file_path)
            output_name = os.path.basename(file_path).replace(ext, '_parsed.txt')
            save_text_output(text, output_name)
            print(f"✅ Saved: {output_name}")
        else:
            print(f"⚠️ Unsupported file type: {file_path}")
    except Exception as e:
        print(f"❌ Failed to parse {file_path}: {e}")

# 🔁 Traverse the raw data folder and parse all files
for root, _, files in os.walk(RAW_DATA_FOLDER):
    for file in files:
        file_path = os.path.join(root, file)
        parse_file(file_path)

print("✅ All files parsed and saved to:", OUTPUT_FOLDER)
