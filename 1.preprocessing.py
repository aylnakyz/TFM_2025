import os
import re
import chardet
import pandas as pd
from collections import Counter
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')

CODES = ['TI', 'REF', 'UI', 'FAM', 'ACT', 'InF']
SPONT_FILES = ['1.txt', '1_1.txt', '1_2.txt', '2.txt', '3.txt', '6.txt']
PICTURE_FILES = ['4-Bridge.txt', '4-Couple.txt', '4-Farm.txt']

PICTURE_TASK_MAP = {
    '4-Bridge.txt': 'picture1',
    '4-Couple.txt': 'picture2',
    '4-Farm.txt': 'picture3'
}

def clean_text(raw):
    r_blocks = re.findall(r'\[R\]:\s*(.*?)(?=\[\w\]:|\Z)', raw, re.DOTALL)
    raw = ' '.join(r_blocks)
    raw = re.sub(r'\*\s*[\w\u00e7\u011f\u0131\u00f6\u015f\u00fc\u00c7\u011e\u0130\u00d6\u015e\u00dc]{1,4}\s*\*', '', raw)
    raw = re.sub(r'\b\w+[-–—]\s*\b', '', raw)  
    raw = re.sub(r'\b\w+[-_]+\w*\b', '', raw)
    raw = re.sub(r'\b(\w)\1{2,}\b', '', raw)
    raw = re.sub(r'\[\d{1,2}:\d{2}:\d{2}\]', '', raw)
    raw = re.sub(r'\s+', ' ', raw).strip()
    return raw

def count_codes_in_sentences(sentences):
    code_counts = []
    for sent in sentences:
        counts = Counter()
        for code in CODES:
            pattern = rf'\{{.*?\}}{code}'
            counts[code] = len(re.findall(pattern, sent))
        code_counts.append(counts)
    return code_counts

def split_sentences(text):
    return sent_tokenize(text, language='turkish')

def sliding_window_counts(code_counts, window_size):
    results = []
    for i in range(0, len(code_counts) - window_size + 1):
        window = code_counts[i:i + window_size]
        combined = Counter()
        for c in window:
            combined.update(c)
        row = {code: combined.get(code, 0) for code in CODES}
        row['window_index'] = i
        results.append(row)
    return results

def read_file_with_encoding_detection(filepath):
    with open(filepath, 'rb') as f:
        raw = f.read()
    encoding = chardet.detect(raw)['encoding']
    if encoding is None:
        return ''
    try:
        return raw.decode(encoding, errors='ignore')
    except:
        return ''

def process_data(data_folder):
    os.makedirs('cleaned_texts', exist_ok=True)
    all_rows, all_cleaned_texts, participants = [], [], set()
    total_sentences, total_words = 0, 0
    
    text_stats = []

    def process_text(text, group, participant_id, task_type, out_name, window_size):
        nonlocal total_sentences, total_words
        cleaned = clean_text(text)
        if not cleaned.strip():
            return
        sentences = split_sentences(cleaned)
        if len(sentences) < window_size:
            return
        code_counts = count_codes_in_sentences(sentences)
        windows = sliding_window_counts(code_counts, window_size=window_size)
        for row in windows:
            row.update({'group': group, 'participant': participant_id, 'task': task_type})
            all_rows.append(row)
        path = os.path.join('cleaned_texts', f"{participant_id}_{out_name}.txt")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        all_cleaned_texts.append((participant_id, task_type, group, cleaned))
        total_sentences += len(sentences)
        total_words += len(cleaned.split())

        text_stats.append({
            'file': PICTURE_TASK_MAP.get(out_name + '.txt', out_name),
            'participant': participant_id,
            'group': group,
            'task': task_type,
            'num_words': len(cleaned.split()),
            'num_sentences': len(sentences),
            'num_windows': len(windows)
        })

    for group in ['HC', 'SZH']:
        group_path = os.path.join(data_folder, group)
        for participant_folder in os.listdir(group_path):
            participant_path = os.path.join(group_path, participant_folder)
            if not os.path.isdir(participant_path):
                continue
            match = re.search(r'\d+', participant_folder)
            participant_id = match.group(0).zfill(2) if match else '00'
            participants.add(participant_id)

            combined_spont_text = ''
            for fname in SPONT_FILES:
                fpath = os.path.join(participant_path, fname)
                if os.path.exists(fpath):
                    combined_spont_text += '\n' + read_file_with_encoding_detection(fpath)
            if combined_spont_text.strip():
                process_text(combined_spont_text, group, participant_id, 'spontaneous', 'spontaneous', window_size=5)

            for fname in PICTURE_FILES:
                fpath = os.path.join(participant_path, fname)
                if os.path.exists(fpath):
                    text = read_file_with_encoding_detection(fpath)
                    fname_base = fname.replace('.txt', '')
                    task_label = PICTURE_TASK_MAP.get(fname, 'picture')
                    process_text(text, group, participant_id, task_label, fname_base, window_size=1)

    pd.DataFrame(all_rows).to_csv("sliding_window_analysis.csv", index=False)

    annotation_records = []
    for pid, task, group, text in all_cleaned_texts:
        counts = {code: len(re.findall(rf'\{{.*?\}}{code}', text)) for code in CODES}
        counts.update({'group': group, 'task': task, 'participant': pid})
        annotation_records.append(counts)

    annotation_df = pd.DataFrame(annotation_records)
    raw_summary = annotation_df.groupby(['group', 'task'])[CODES].sum().reset_index()
    global_summary = pd.DataFrame([{
        'total_participants': len(participants),
        'total_sentences': total_sentences,
        'total_words': total_words,
        'total_annotations': annotation_df[CODES].sum().sum()
    }])

    with open("data_description.csv", 'w', encoding='utf-8') as f:
        f.write("### Global Summary\n")
        global_summary.to_csv(f, index=False)
        f.write("\n### Raw Annotation Counts by Group and Task (from cleaned_texts)\n")
        raw_summary.to_csv(f, index=False)

    
      

    text_stats_df = pd.DataFrame(text_stats)
    annotation_df_full = pd.DataFrame(annotation_records)
    text_summary = pd.merge(text_stats_df, annotation_df_full, on=['participant', 'group', 'task'])
    text_summary.to_csv("text_summary.csv", index=False)

    print("Preprocessing completed.")

if __name__ == "__main__":
    process_data("data")
