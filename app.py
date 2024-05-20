import pandas as pd
import spacy as sp
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter
import re
from scipy.stats import pearsonr
from nltk import ngrams
import requests
import json
from requests.sessions import Session
from nltk.tokenize import word_tokenize
import re
from flask import Flask, render_template, request
import matplotlib
from flask import Flask, request, render_template, redirect, url_for, flash,jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask import render_template, request, url_for, redirect
from datetime import datetime
import time
from datetime import datetime, timezone
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from io import BytesIO
import base64
import pandas as pd
from collections import Counter
import spacy

import requests
import json
from concurrent.futures import ThreadPoolExecutor
#cleaning raw text file
def clean(text):
    cleaned_text = text.replace('\t', '').replace('\n', '').replace(',', '')
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\d+', '', cleaned_text)
    cleaned_text = re.sub("(http\S+)", "", cleaned_text)
    cleaned_text = cleaned_text.replace(' .', '.').replace('. ', '.')
    cleaned_text = re.sub(r'[^\w\s.]', '', cleaned_text)
    return cleaned_text
#sentence segmenta
def sentence(text):
    sentences = []
    current = ""
    
    for i in range(len(text)):
        current += text[i]
        if text[i] == '.' and i < len(text) - 1 and text[i-2] != ' ' and text[i-1] != '.':
            if current.strip() and ' ' in current.strip():
                sentences.append(current.strip())
                current = ''
    if current.strip() and ' ' in current.strip():
        sentences.append(current.strip())
    df = pd.DataFrame({'Sentence': sentences})
    return df

def remove_tabs_and_newlines(column):
    return column.apply(lambda x: re.sub(r'[^\w\s.@]', '', x))

def remove_mail_and_fraction(s):
    while '@' in s:
        at_index = s.find('@')
        left = at_index - 1
        right = at_index + 1
        while left >= 0 and s[left] != ' ':
            left -= 1
        while right < len(s) and s[right] != ' ':
            right += 1

        s = s[:left+1] + s[right:]
    dot_index = s.find('.')
    while dot_index != -1:
        left = dot_index - 1
        right = dot_index + 1
        if (left >= 0 and s[left].isdigit()) or (right < len(s) and s[right].isdigit()):
            while left >= 0 and s[left].isdigit() and s[left] != ' ':
                left -= 1
            while right < len(s) and s[right].isdigit() and s[right] != ' ':
                right += 1

            s = s[:left+1] + s[right:]

        dot_index = s.find('.', dot_index + 1)

    return s




session = requests.Session()
def remove_words(text_list, remove_list):
    return [' '.join(word for word in text.split() if word not in remove_list) for text in text_list]


def process_json_text(txt_in, retry=0):
    base_url = 'http://64.119.31.61:8080/nlp-web-demo/process?text='
    max_retries = 2  
    
    try:
        res = session.get(base_url + str(txt_in), timeout=3)
        res.raise_for_status()
        modified_text = res.text.replace('"stopWordType":"', '"stopWordType":"# ')
        parsed_data = json.loads(modified_text)
        extracted_values = [
            ' '.join(
                value for item in inner_list for value in item.values()
                if value not in ['"', '{', '}', 'word', 'lemma', 'posTag', 'nameType', 'stopWordType', ',', ':']
            )
            for inner_list in parsed_data
        ]
        return '\n'.join(extracted_values)

    except (requests.RequestException, TimeoutError) as e:
        if retry < max_retries:
            return process_json_text(txt_in, retry + 1)
        else:
            return 'Failed to process text after several attempts.'

"""
def process_json_text(txt_in):
    base_url = 'http://64.119.31.61:8080/nlp-web-demo/process?text='
    try:
        # Use the persistent session to make requests
        res = session.get(base_url + str(txt_in))
        res.raise_for_status()  # Checks if request was successful
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return ''

    # Process response text directly if the status is OK
    modified_text = res.text.replace('"stopWordType":"', '"stopWordType":"# ')
    parsed_data = json.loads(modified_text)
    extracted_values = [
        ' '.join(
            value for item in inner_list for value in item.values()
            if value not in ['"', '{', '}', 'word', 'lemma', 'posTag', 'nameType', 'stopWordType', ',', ':']
        )
        for inner_list in parsed_data
    ]
    return '\n'.join(extracted_values)
"""


# Example usage:
# result = process_json_text("your input text here")

"""
def process_dataframe(df):
    processed_results = []
    for sentence in df['Sentence']:
        if isinstance(sentence, list):
            sentence = ' '.join(sentence)  # Convert list of strings to a single string
        processed_result = process_json_text(sentence)
        processed_results.append(processed_result)
    
    df['ProcessedResult'] = processed_results
    return df

from concurrent.futures import ThreadPoolExecutor
"""
def process_dataframe(df):
    # Helper function to process each sentence
    def process_each(sentence):
        if isinstance(sentence, list):
            sentence = ' '.join(sentence)  # Convert list of strings to a single string
        return process_json_text(sentence)
    
    # Create a list of sentences
    sentences = df['Sentence'].tolist()
    
    # Process sentences in parallel
    with ThreadPoolExecutor(max_workers=6) as executor:
        processed_results = list(executor.map(process_each, sentences))
    
    # Store results back in the dataframe
    df['ProcessedResult'] = processed_results
    return df


###
# Example usage:
# df = pd.DataFrame({'Sentence': ['sentence 1', 'sentence 2', 'sentence 3']})
# processed_df = process_dataframe(df)
# print(processed_df)


#web-r bolovsruulsan ugeer ni salgah

def word(text):
    result = []
    sentences = ""
    current = ""
    i = 0
    
    while i < len(text):
        if text[i] == '[':
            while i+1 < len(text) and (text[i+1] != '+' and text[i+1] != ']'):
                i += 1
                current += text[i]
            # Check if there is a '#' after the current position
            while i+1 < len(text) and text[i] != '#':
                i += 1
            # Check if the word is followed by "# None", if yes, add to result
            if i+6 < len(text) and text[i+1:i+6] == " None":
                sentences = sentences + current + ' '
            current = ''
        i += 1
    result.append(sentences.strip())
    return result


def word_dataframe(df):
    df['word'] = df['ProcessedResult'].apply(word)
    return df


def clean1(words_list):
    cleaned_words = []
    for word in words_list:
        cleaned_word = str(word).replace('.', '')
        cleaned_words.append(cleaned_word)
    return cleaned_words
#ugiin aimag salgah
def wordca(text):
    result = []
    current = ""

    for i in range(len(text)-1):
        if text[i] == ']' and text[i+1] == ' ':
            i = i + 1 
            while text[i+1] != ' ':
                i = i + 1
                current += text[i]

            result.append(current.strip())
            current = ''

    return result

def wordca_dataframe(df):
    df['wordca'] = df['ProcessedResult'].apply(wordca)
    return df

#nerlesen ugiig salgah 
def wordca1(text):
    result = []
    current = ""
    i = 0
    
    while i < len(text) - 1:
        if text[i] == ']' and text[i + 1] == ' ':
            i = i + 2
            while i < len(text) - 1 and text[i + 1] != ' ':
                i = i + 1
            while i < len(text) - 1 and text[i + 1] == ' ':
                i = i + 1
            while i < len(text) - 1 and text[i + 1] != ' ':
                i = i + 1
                current += text[i]
            if current != 'O':
                result.append(current.strip())
            current = ''

        i = i + 1

    return result

def wordca_dataframe1(df):
    new_df = pd.DataFrame()
    new_df['wordca'] = df['ProcessedResult'].apply(wordca1)
    new_df = new_df[new_df['wordca'].astype(bool)] 
    return new_df
#text-s ug buriin davtamj oloh
def tokenize_and_count(df, column_name='word', tokenized_column_name='Tokenized'):
    df[tokenized_column_name] = df[column_name].astype(str).apply(word_tokenize)
    
    word_counts = Counter()

    for tokens in df[tokenized_column_name]:
        word_counts.update(tokens)

    word_counts_df = pd.DataFrame.from_dict(word_counts, orient='index', columns=['Count']).reset_index()
    word_counts_df.columns = ['Word', 'Count']
    word_counts_df = word_counts_df.sort_values(by='Count', ascending=False)

    return word_counts_df
#nerlsen ug tooloh
def process_and_count_words(df, column_name):
    df[column_name] = df[column_name].apply(
        lambda x: [word.split('-')[1] for word in x if len(word.split('-')) > 1]
    )

    all_words = [word for sublist in df[column_name] for word in sublist]
    word_counts = pd.Series(all_words).value_counts()
    word_counts_df = pd.DataFrame({'Word': word_counts.index, 'Count': word_counts.values})
    word_counts_df = word_counts_df.sort_values(by='Count', ascending=False)

    return word_counts_df
#uguulberiin ug tooloh
def sentence_word_counter(data_frame):
    word_counts_per_sentence = []
    for index, row in data_frame.iterrows():
        sentence_text = ' '.join(row["Tokenized"])
        words = word_tokenize(sentence_text)
        word_count = len(words)
        word_counts_per_sentence.append(word_count)
    data_frame["Word_Counts"] = word_counts_per_sentence

    return data_frame
#ugiin aimag tooloh 
def plot_word_counts(df, column_name, plot_title='Word Counts'):
    if column_name not in df.columns:
        return

    word_counts = Counter()

    for tokens in df[column_name]:
        word_counts.update(tokens)

    word_counts_df = pd.DataFrame.from_dict(word_counts, orient='index', columns=['Count']).reset_index()
    word_counts_df.columns = ['Word', 'Count']
    word_counts_df = word_counts_df.sort_values(by='Count', ascending=False)

    return word_counts_df

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text
    
def find_word_(word, text):
    instances = []
    pattern = re.compile(r'\b{}\b'.format(re.escape(word)))
    for match in re.finditer(pattern, text):
        instances.append(match.start())
    return instances


def find_word(row, text):
    word = row['Word']
    indices = find_word_(word, text)
    return indices

def remove_special_characters(cleaned_text):
    cleaned_text = re.sub(r'\d+', '', cleaned_text)
    cleaned_text = re.sub("(http\S+)", "", cleaned_text)
    cleaned_text = cleaned_text.replace(' .', '.').replace('. ', '.')
    cleaned_text = re.sub(r'[^\w\s.]', ' ', cleaned_text)
    cleaned_text = cleaned_text.replace('\t', '').replace('\n', '').replace(',', '')
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text

def find_word_counter(loc, text):
    result = []
    sentences = ""
    current = ""
    
    for i in loc:
        while i >= 0 and text[i] != '[':
            i = i - 1
        while i < len(text) and text[i] != ']':
            current += text[i]
            i = i + 1
        current = remove_special_characters(current) 
        sentences = sentences + current + ' '
        current = ''
    
    result.append(sentences.strip())
    return result

def apply_find_word_counter(df, text):
    df['words'] = df['wordloc'].apply(find_word_counter, text=text)
    return df

#mur buriin ugiin aimgiin huviral tooloh
def tokenize_and_count_single_row(row, column_name='words'):
    tokens = word_tokenize(str(row[column_name]))
    word_counts = Counter(tokens)
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    clean_sorted_word_counts = [(re.sub(r'\W+', '', word), count) for word, count in sorted_word_counts]
    word_counts_str = ', '.join([f"{word}: {count}" for word, count in clean_sorted_word_counts])

    return word_counts_str



def process(file_content, result_df1, result_df2, result_df,result_df_):
    text1 = remove_mail_and_fraction(file_content)
    text1 = clean(text1)
    df = sentence(text1)
    df = df.apply(remove_tabs_and_newlines)
    df['Sentence'] = df['Sentence'].apply(lambda x: sent_tokenize(x))

    df = process_dataframe(df)

    df = word_dataframe(df)
    df['word'] = df['word'].apply(clean1)
    df['word'] = df['word'].apply(lambda x: x[0] if x else '')
    df = wordca_dataframe(df)
    df['Tokenized'] = df['word'].apply(word_tokenize)
    df = sentence_word_counter(df)
    df = sentence_word_counter_(df)
    result_df1 = tokenize_and_count(df, column_name='word')

    result_df = wordca_dataframe1(df)
    

    result_df2 = process_and_count_words(result_df, column_name='wordca')

    result_df_ = plot_word_counts(df, column_name='wordca', plot_title='')
    text = df['ProcessedResult'].str.cat(sep=' ')
    result_df_['wordloc'] = result_df_.apply(find_word, text=text, axis=1)
    result_df_ = apply_find_word_counter(result_df_, text=text)
    result_df_['count'] = result_df_.apply(tokenize_and_count_single_row, axis=1)
    
    df = wordca_dataframe(df)
    return df, result_df1, result_df2, result_df, result_df_





app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")


def plot_word_frequency(df):
    plt.figure(figsize=(10, 6))
    plt.bar(df["word"], df["Word_Counts"])
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.title("Word Frequency Distribution")
    plt.xticks(rotation=45)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode("utf-8")
    return plot_url


def process_file(text_data):
    """
    Calculate word lengths and return a DataFrame containing word length frequencies.
    """
    word_len = [len(word) for word in text_data.split()]
    word_len_s = set(word_len)
    
    word_len_count = [word_len.count(length) for length in word_len_s]
    
    data = {
        "word_len": list(word_len_s),
        "frequency": word_len_count
    }
    
    return pd.DataFrame(data), len(word_len)


def main1(text_data1, text_data2):
    """
    Process two texts and return DataFrames for each.
    """
    data1, len1 = process_file(text_data1)
    data2, len2 = process_file(text_data2)
    data1 = data1.sort_values(by='word_len').reset_index(drop=True)
    data2 = data2.sort_values(by='word_len').reset_index(drop=True)
    return data1, data2, len1, len2

def sentence_word_counter_(data_frame):
    word_counts_per_sentence = []
    for index, row in data_frame.iterrows():
        sentence_text = ' '.join(row["Sentence"])
        words = word_tokenize(sentence_text)
        word_count = len(words)
        word_counts_per_sentence.append(word_count)
    data_frame["Word_Counts_"] = word_counts_per_sentence

    return data_frame
def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def generate_ngrams(text, n):
    tokens = text.split()
    return list(ngrams(tokens, n))

# Function to calculate ngram frequency and rank
def calculate_ngram_frequency(text, n, top_n=None):
    ngram_freq = Counter()
    
    # Generate ngrams for the text
    ngrams_list = generate_ngrams(text, n)
    ngram_freq.update(ngrams_list)
    ngram_df = pd.DataFrame.from_dict(ngram_freq, orient='index', columns=['Давтамж'])
    ngram_df.index = ngram_df.index.map(' '.join)
    ngram_df.index.name = 'Үг' if n == 1 else f'{n}-gram'
    ngram_df = ngram_df.sort_values(by='Давтамж', ascending=False)
    ngram_df['Үг'] = ngram_df.index
    ngram_df.reset_index(drop=True, inplace=True)
    if top_n is not None:
        ngram_df = ngram_df.head(top_n)
    
    return ngram_df
def word_cluster(sentence, n, selected_word):
    words = sentence.split()
    left_clusters = []
    right_clusters = []

    for i, word in enumerate(words):
        if word == selected_word:
            context_start = max(0, i - n)
            context_end = min(len(words), i + n + 1)
            left_context = ' '.join(words[context_start:i])
            right_context = ' '.join(words[i+1:context_end])
            left_clusters.append({'Зүүн': left_context, 'Word': selected_word})
            right_clusters.append({'Баруун': right_context, 'Word': selected_word})

    return pd.DataFrame(left_clusters), pd.DataFrame(right_clusters)

def word_cluster_dataframe(dataframe, columns, n, selected_word):
    left_result = pd.DataFrame(columns=['Зүүн', 'Word'])
    right_result = pd.DataFrame(columns=['Word', 'Баруун'])

    for col in columns:
        for sentence_list in dataframe[col]:  
            for sentence in sentence_list:    
                left_cluster, right_cluster = word_cluster(sentence, n, selected_word)
                left_result = pd.concat([left_result, left_cluster], ignore_index=True)
                right_result = pd.concat([right_result, right_cluster], ignore_index=True)

    return left_result, right_result

def process_texts(text1, text2):
    result_df1 = pd.DataFrame()
    result_df2 = pd.DataFrame()
    result_df = pd.DataFrame()
    result_df_ = pd.DataFrame()
    df, result_df1, result_df2, result_df, result_df_ = process(text1, result_df1, result_df2, result_df, result_df_)

    result_df12 = pd.DataFrame()
    result_df22 = pd.DataFrame()
    result_df32 = pd.DataFrame()
    result_df_2 = pd.DataFrame()
    # Process the second text and update the dataframes
    df2, result_df12, result_df22, result_df32, result_df_2 = process(text2, result_df12, result_df22, result_df32, result_df_2)

    df1_subset = result_df_.iloc[:, :2]
    df2_subset = result_df_2.iloc[:, :2]
    merged_df = pd.merge(df1_subset, df2_subset, on='Word', how='outer')
    merged_df = merged_df.sort_values(by='Count_x', ascending=False).fillna(0)
    merged_df = merged_df.head(30)
    merged_df1 = merged_df
    length1 = len(df)
    length2 = len(df2)

    # First Plot: Word Frequency Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(merged_df['Word'], merged_df['Count_x']/length1, marker='o', linestyle='-', color='red', label='Text 1')
    plt.plot(merged_df['Word'], merged_df['Count_y']/length2, marker='o', linestyle='-', color='blue', label='Text 2')
    
    plt.xlabel('Үгийн аймаг', fontsize=14)
    plt.ylabel('Давтамж', fontsize=14)
    plt.title('Өгүүлбэр дэх үгийн аймгийн давтамжийн тархалтын харьцуулалт', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    correlation_coefficient, _ = pearsonr(merged_df['Count_x']/length1, merged_df['Count_y']/length2)
    plt.text(0.7, 0.9, f'Төсөөгийн коеф: {correlation_coefficient:.2f}', transform=plt.gca().transAxes)

    buf1 = BytesIO()
    plt.savefig(buf1, format="png")
    buf1.seek(0)
    plot_url1 = base64.b64encode(buf1.getvalue()).decode("utf-8")
    plt.close()

    # Second Plot: Sentence Word Count Comparison
    plt.figure(figsize=(12, 6))
    sorted_counts1 = df['Word_Counts'].sort_values(ascending=False)
    plt.plot(df.index, sorted_counts1, linestyle='-', color='red', marker='o', label='Text 1')
    sorted_counts2 = df2['Word_Counts'].sort_values(ascending=False)
    plt.plot(df2.index, sorted_counts2, linestyle='-', color='blue', marker='o', label='Text 2')
    
    plt.xlabel('Өгүүлбэр', fontsize=14)
    plt.ylabel('Өгүүлбэр дэх үгийн тоо', fontsize=14)
    plt.title('Өгүүлбэрийн дэх үгийн давтамжийн харьцуулалт', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    buf1 = BytesIO()
    plt.savefig(buf1, format="png")
    buf1.seek(0)
    plot_url2 = base64.b64encode(buf1.getvalue()).decode("utf-8")
    plt.close()

    # Third Plot: Word Count Comparison
    min_length = min(len(result_df1), len(result_df12))
    min_length = min(min_length, 30)
    result_df1 = result_df1.iloc[:min_length]
    result_df12 = result_df12.iloc[:min_length]
    sum1 = result_df1['Count'].sum()
    sum2 = result_df12['Count'].sum()

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(result_df1)), result_df1['Count']/sum1, linestyle='-', color='red', marker='o', label='Text 1')
    plt.plot(range(len(result_df12)), result_df12['Count']/sum2, linestyle='-', color='blue', marker='o', label='Text 2')
    
    correlation_coefficient, _ = pearsonr(result_df1['Count'], result_df12['Count'])
    plt.text(0.6, 0.9, f'Төсөөгийн коеф: {correlation_coefficient:.2f}', transform=plt.gca().transAxes)
    plt.xlabel('Үг', fontsize=14)
    plt.ylabel('Үгийн давьтамжийн нийт үгт харьцуулалт', fontsize=14)
    plt.title('Үгийн давтамжийн тархалтын харьцуулалт', fontsize=16)
    plt.xticks([])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    buf1 = BytesIO()
    plt.savefig(buf1, format="png")
    buf1.seek(0)
    plot_url3 = base64.b64encode(buf1.getvalue()).decode("utf-8")
    plt.close()

    # Fourth Plot: Word Length Frequency Comparison
    text12 = ' '.join(flatten_list(df['Sentence']))
    text13 = ' '.join(flatten_list(df2['Sentence']))
    data1, data2, len1, len2 = main1(text12, text13)
    
    plt.figure(figsize=(12, 6))
    merged_df = pd.merge(data1, data2, on='word_len', how='outer').fillna(0)
    merged_df = merged_df.sort_values(by='word_len')
    
    plt.plot(merged_df['word_len'], merged_df['frequency_x']/len1, marker='o', linestyle='-', color='red', label='Text 1')
    plt.plot(merged_df['word_len'], merged_df['frequency_y']/len2, marker='o', linestyle='-', color='blue', label='Text 2')
    
    plt.xlabel('Үгийн урт', fontsize=14)
    plt.ylabel('Үгийн уртын давтамж', fontsize=14)
    plt.title('Үгийн уртын давтамжийн тархалтын харьцуулалт', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    correlation_coefficient, _ = pearsonr(merged_df['frequency_x'], merged_df['frequency_y'])
    plt.text(0.6, 0.9, f'Төсөөгийн коеф: {correlation_coefficient:.2f}', transform=plt.gca().transAxes)

    buf1 = BytesIO()
    plt.savefig(buf1, format="png")
    buf1.seek(0)
    plot_url4 = base64.b64encode(buf1.getvalue()).decode("utf-8")
    plt.close()

    # N-gram Frequencies
    df = df.drop(columns=['wordca', 'ProcessedResult', 'Tokenized'])
    df2 = df2.drop(columns=['wordca', 'ProcessedResult', 'Tokenized'])
    text12 = ''.join(flatten_list(df['word']))
    text13 = ''.join(flatten_list(df2['word']))
    
    result_df11 = calculate_ngram_frequency(text12, 1, top_n=30)
    result_df21 = calculate_ngram_frequency(text13, 1, top_n=30)
    result_df12 = calculate_ngram_frequency(text12, 2, top_n=30)
    result_df22 = calculate_ngram_frequency(text13, 2, top_n=30)
    result_df23 = calculate_ngram_frequency(text12, 3, top_n=30)
    result_df13 = calculate_ngram_frequency(text13, 3, top_n=30)
    
    return df, df2, result_df11, result_df12, result_df13, result_df21, result_df22, result_df23, plot_url1, plot_url2, plot_url3, plot_url4, merged_df1


def ngramplot(df, n):
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df["Үг"], df["Давтамж"], color='skyblue', label='Text 1', edgecolor='black')
    
    plt.xlabel("Үг", fontsize=14)
    plt.ylabel("Давтамж", fontsize=14)
    plt.title(f"{n}-Үгийн давтамжийн тархалт", fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)  
    plt.yticks(fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, int(yval), ha='center', va='bottom', fontsize=10, color='black')

    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return plot_url
def process_text(text, word):
    result_df1 = pd.DataFrame()
    result_df2 = pd.DataFrame()
    result_df = pd.DataFrame()
    result_df_ = pd.DataFrame()
    df, result_df1, result_df2, result_df, result_df_ = process(text, result_df1, result_df2, result_df, result_df_)
    
    if len(df) > 1000:
        agg_counts1 = df['Word_Counts'].groupby(df.index // 100).mean()
        agg_counts2 = df['Word_Counts_'].groupby(df.index // 100).mean()

        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(len(agg_counts1)), agg_counts1, linestyle='-', color='red', marker='o', label='Өмнө нь')
        plt.plot(np.arange(len(agg_counts2)), agg_counts2, linestyle='-', color='blue', marker='o', label='Дараа нь')
    else:
        plt.figure(figsize=(12, 6))
        sorted_counts1 = df['Word_Counts'].sort_values(ascending=False)
        plt.plot(df.index, sorted_counts1, linestyle='-', color='red', marker='o', label='Өмнө нь')
        sorted_counts2 = df['Word_Counts_'].sort_values(ascending=False)
        plt.plot(df.index, sorted_counts2, linestyle='-', color='blue', marker='o', label='Дараа нь')

    plt.xlabel('Өгүүлбэрийн дэс дугаар', fontsize=14)
    plt.ylabel('Үгийн тоо', fontsize=14)
    plt.title('Сул үг хассаны өмнөх болон дараах өгүүлбэр дэх үгийн тоо', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    buf1 = BytesIO()
    plt.savefig(buf1, format="png")
    buf1.seek(0)
    plot_url = base64.b64encode(buf1.getvalue()).decode("utf-8")
    plt.close()

    text12 = ' '.join(flatten_list(df['Sentence']))
    text13 = ''.join(flatten_list(df['word'])) 
    data1, data2, len1, len2 = main1(text12, text13)
    
    plt.figure(figsize=(12, 6))
    merged_df = pd.merge(data1, data2, on='word_len', how='outer').fillna(0)
    merged_df = merged_df.sort_values(by='word_len')
    
    plt.plot(merged_df['word_len'], merged_df['frequency_x'], marker='o', linestyle='-', color='red', label='Өмнө нь')
    plt.plot(merged_df['word_len'], merged_df['frequency_y'], marker='o', linestyle='-', color='blue', label='Дараа нь')
    
    plt.xlabel('Үгийн урт', fontsize=14)
    plt.ylabel('Үгийн уртын давтамж', fontsize=14)
    plt.title('Үгийн язгуураар салгасны өмнөх болон дараах үгийн уртын тархалт', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    buf1 = BytesIO()
    plt.savefig(buf1, format="png")
    buf1.seek(0)
    plot_url1 = base64.b64encode(buf1.getvalue()).decode("utf-8")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, sorted_counts1, linestyle='-', color='red', marker='o', label='text 1')

    plt.xlabel('Өгүүлбэр', fontsize=14)
    plt.ylabel('Өгүүлбэр дэх үгийн тоо', fontsize=14)
    plt.title('Өгүүлбэрийн дэх үгийн давтамжийн харьцуулалт', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    buf1 = BytesIO()
    plt.savefig(buf1, format="png")
    buf1.seek(0)
    plot_url21 = base64.b64encode(buf1.getvalue()).decode("utf-8")
    plt.close()

    df = df.drop(columns=['wordca', 'ProcessedResult', 'Tokenized'])
    gram_df2 = calculate_ngram_frequency(text13, 2, top_n=30)
    gram_df3 = calculate_ngram_frequency(text13, 3, top_n=30)
    gram_df1 = calculate_ngram_frequency(text13, 1, top_n=30)
    top_ngrams = gram_df1.head(15)
    columns_to_apply = ['Sentence']
    n = 3  
    left_contexts, right_contexts = word_cluster_dataframe(df, columns_to_apply, n, word)
    merged_contexts = pd.concat([left_contexts, right_contexts['Баруун']], axis=1)
    plot_url11 = ngramplot(gram_df1, 1)
    plot_url12 = ngramplot(gram_df2, 2)
    plot_url13 = ngramplot(gram_df3, 3)
    
    return df, gram_df1, gram_df2, gram_df3, df, plot_url, plot_url1, plot_url11, plot_url12, plot_url13, plot_url21

def kwic(sentence, n, word):
    if not isinstance(sentence, str):
        sentence = str(sentence)
    
    words = sentence.split()
    try:
        index = words.index(word)
    except ValueError:
        return None  

    left_start = max(0, index - n)
    right_end = min(len(words), index + n + 1)
    left_context = ' '.join(words[left_start:index])
    right_context = ' '.join(words[index+1:right_end])
    return pd.DataFrame({
        'Left Context': [left_context],
        'Word': [word],
        'Right Context': [right_context]
    })


def kwic_dataframe(dataframe, columns, n, word):
    result_rows = []
    
    for index, row in dataframe.iterrows():
        for col in columns:
            sentence = row[col]
            if pd.notna(sentence):  
                kwic_result = kwic(sentence, n, word)
                if kwic_result is not None:
                    result_rows.append(kwic_result)

    if result_rows:
        return pd.concat(result_rows, ignore_index=True)
    else:
        return pd.DataFrame(columns=['Left Context', 'Word', 'Right Context'])  

from gensim import corpora, models
import gensim
def extract_topics(lda_model, num_words=5):
    topics = []
    for topic_id, topic in lda_model.print_topics(num_topics=-1, num_words=num_words):
        word_probs = []
        for word, prob in lda_model.show_topic(topic_id, topn=num_words):
            word_probs.append((word, round(prob, 4)))  
        topics.append({
            'topic_id': topic_id,
            'words': word_probs
        })
    return topics

def perform_topic_modeling(texts, num_topics=5, num_words=5):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    
    topics = extract_topics(lda_model, num_words)
    return topics


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SECRET_KEY'] = 'your-secret-key'
db = SQLAlchemy(app)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
login_manager = LoginManager(app)

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    text_input1 = db.Column(db.Text, nullable=False)
    text_input2 = db.Column(db.Text, nullable=True)  
    status = db.Column(db.String(50), default='pending') 
    result_html = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    user = db.relationship('User', backref=db.backref('tasks', lazy=True))

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

class HtmlContent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    user = db.relationship('User', backref=db.backref('html_contents', lazy=True))


from threading import Thread

def task_checker():
    with app.app_context():  
        while True:
            time.sleep(10)
            process_tasks()

        
def start_background_task():
    thread = Thread(target=task_checker)
    thread.daemon = True  
    thread.start()


with app.app_context():
    db.create_all()
    start_background_task()

login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route("/", methods=["GET", "POST"])
def index():
    text1 = "" 
    if request.method == "POST":
        if 'file1' in request.files:
            file = request.files['file1']
            if file.filename != '':
                text = file.read().decode("utf-8")
        if 'file2' in request.files:
            file = request.files['file2']
            if file.filename != '':
                text1 = file.read().decode("utf-8") 
        else:
            text = request.form.get("text1", "")
            text1 = request.form.get("text2", "")
        if request.form.get("action") == "process_word":
            return process_data(request)
        if text1:  
            result_html = process_dual_texts(text, text1)
            save_results(current_user.id, result_html)
            return result_html
        else:  
            result_html = process_single_text(text)
            save_results(current_user.id, result_html)
            return result_html
        
    return render_template("index.html", user=current_user)


@app.route('/save_task', methods=['POST'])
@login_required
def save_task():
    data = request.get_json()
    text = data.get('text', '')
    text1 = data.get('text1', '')
    new_task = Task(user_id=current_user.id, text_input1=text, text_input2=text1 if text1 else None)
    db.session.add(new_task)
    db.session.commit()
    return jsonify({"status": "success"})

def process_tasks():
    pending_tasks = Task.query.filter_by(status='pending').all()
    for task in pending_tasks:
        start_time = time.time()  
        try:
            task.status = 'processing'
            db.session.commit()

            if task.text_input2:
                result_html = process_dual_texts(task.text_input1, task.text_input2)
            else:
                result_html = process_single_text(task.text_input1)
            save_results(task.user_id, result_html)
            db.session.delete(task)
            db.session.commit()

        except Exception as e:
            task.status = 'error'
            db.session.commit()
            print(f"Error processing task {task.id}: {e}")



def remove_words(text_list, remove_list):
    return [word for word in text_list if word not in remove_list]
def process_single_text(text):
    df,result_df1,result_df2,result_df3, df,plot_url, plot_url1,plot_url11,plot_url12, plot_url13, plot_url21 = process_text(text, word)
    text2 = ''.join(flatten_list(df['word']))
    words = text2.split()  
    words1 = text.split()  
    word_count1 = len(words)  
    word_count = len(words1) 
    height = df.shape[0]
    
    words = text2.split()  

    words_to_remove = ['байна', 'байгаа', '[байлаа', 'байв' , 'байлаа', 'байдаг', 'байх', 'бай', 'бол']
    words = remove_words(words, words_to_remove)
    texts = [words]
    topic = perform_topic_modeling(texts)
    
    if height > 300:
        df1 = df.head(300)
    if len(text) > 10000:
        text1 = text[:10000]
    if len(text2) > 10000:
        text21 = text2[:10000]
    text1 = text
    text21 = text2
    df1 = df
    result_html = render_template("single_result.html", height = height, text=text1,word_count=word_count, word_count1=word_count1, result_df1=result_df1.to_html(index=False), plot_url=plot_url, plot_url1=plot_url1, text2=text21, result_df2=result_df2.to_html(index=False), result_df3=result_df3.to_html(index=False), df=df1.to_html(index=False), plot_url11=plot_url11, plot_url12=plot_url12, plot_url13=plot_url13, topic = topic, plot_url21=plot_url21)  # Assuming 'context' contains all the data you pass to the template
    df.to_csv('path_to_save_dataframe.csv', index=False)
    return result_html


def process_dual_texts(text, text1):
    df, df2, result_df11, result_df12, result_df13, result_df21, result_df22, result_df23,plot_url1, plot_url2, plot_url3, plot_url4, merged_df = process_texts(text, text1)
    text3 = ''.join(flatten_list(df['word']))
    text4 = ''.join(flatten_list(df2['word']))
    df = df.drop(columns=[ 'Word_Counts'])
    df2 = df2.drop(columns=[ 'Word_Counts'])
    words = text3.split()  
    word_count1 = len(words) 
    words = text4.split()  
    word_count2 = len(words)  
    words = text.split()  
    word_count = len(words) 
    words = text1.split() 
    word_count_ = len(words)  
    height = df.shape[0]
    height1 = df2.shape[0]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    similarity = round(similarity, 3)
    tfidf = vectorizer.fit_transform([text3, text4])
    similarity1 = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    similarity1 = similarity1.round(3)
    if height > 300:
        df = df.head(300)
    if height1 > 300:
        df2 = df2.head(300)
    if len(text1) > 10000:
        text1 = text1[:10000]
    if len(text) > 10000:
        text = text[:10000]
    if len(text3) > 10000:
        text3 = text3[:10000]
    if len(text4) > 10000:
        text4 = text4[:10000]
    result_html = render_template("result.html", text=text, text1=text1, text3=text3, text4=text4, height = height, height1 = height1,
                                   word_count1=word_count1, word_count2=word_count2, word_count=word_count, word_count_=word_count_, 
                                   df=df.to_html(index=False), result_df11=result_df11.to_html(index=False),result_df12=result_df12.to_html(index=False),result_df13=result_df13.to_html(index=False),
                                   result_df21=result_df21.to_html(index=False), result_df22=result_df22.to_html(index=False),result_df23=result_df23.to_html(index=False), merged_df=merged_df.to_html(index=False),
                                   df2=df2.to_html(index=False),
                                   plot_url2=plot_url2, plot_url1=plot_url1, plot_url3=plot_url3, plot_url4=plot_url4, similarity=similarity,similarity1=similarity1)  # Assuming 'context' contains all the data you pass to the template
    return result_html



@app.route("/process-data", methods=["POST"])
def process_data():
    df = pd.read_csv('path_to_save_dataframe.csv')
    word = request.json['text']
    kwic_df = kwic_dataframe(df, ['word'], 3, word)
    return jsonify({'html': kwic_df.to_html(classes='myDataFrameStyle', index=False)})

def save_results(user_id, html_content):
    new_html_record = HtmlContent(user_id=user_id, content=html_content)
    db.session.add(new_html_record)
    db.session.commit()

@app.route('/history')
@login_required
def history():
    sessions = HtmlContent.query.filter_by(user_id=current_user.id)\
                                .order_by(HtmlContent.created_at.desc())\
                                .limit(10)\
                                .all()
    return render_template('history.html', html_contents=sessions)

@app.route('/view-session/<int:session_id>')
@login_required
def view_session(session_id):
    html_record = HtmlContent.query.get_or_404(session_id)
    return html_record.content  

"""
@app.route("/process-data", methods=["POST"])
def process_data():
    df = pd.read_csv('path_to_save_dataframe.csv')
    word = request.json['text']
    kwic_df = kwic_dataframe(df, ['word'], 3, word)
    return jsonify({'html': kwic_df.to_html(classes='myDataFrameStyle')})
"""
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))
@app.route('/clear-history', methods=['POST'])
@login_required
def clear_history():
    try:
        HtmlContent.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        flash('History cleared successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Failed to clear history.', 'error')
    return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(debug=True, port=5008)