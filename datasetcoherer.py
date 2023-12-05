import os
import tiktoken
import argparse
import json
from tqdm import tqdm
import re
import cohere 
from fitzcli import *

# Constants
TOKEN_CHUNK_SIZE = 300
EMBEDDING_ENCODING = 'cl100k_base'

co = cohere.Client('MfkoPnOYfdjb9hln30HfWlvgcCYuf0TAeQ2zOx0g')  # Replace with your actual Cohere API key



class PDFExtractor:
    def __init__(self, input_pdf, output_txt, pages="1-N", mode="layout", grid=1, fontsize=3, noformfeed=False, skip_empty=True, convert_white=False, noligatures=True, extra_spaces=False):
        self.input_pdf = input_pdf
        self.output_txt = output_txt
        self.pages = pages
        self.mode = mode
        self.grid = grid
        self.fontsize = fontsize
        self.noformfeed = noformfeed
        self.skip_empty = skip_empty
        self.convert_white = convert_white
        self.noligatures = noligatures
        self.extra_spaces = extra_spaces

    def extract_text(self):
        doc = open_file(self.input_pdf, password=None, pdf=False)
        pagel = get_list(self.pages, doc.page_count + 1)
        output = self.output_txt
        textout = open(output, "wb")
        flags = TEXT_PRESERVE_LIGATURES | TEXT_PRESERVE_WHITESPACE
        if self.convert_white:
            flags ^= TEXT_PRESERVE_WHITESPACE
        if self.noligatures:
            flags ^= TEXT_PRESERVE_LIGATURES
        if self.extra_spaces:
            flags ^= TEXT_INHIBIT_SPACES
        func = {
            "simple": page_simple,
            "blocks": page_blocksort,
            "layout": page_layout,
        }
        for pno in pagel:
            page = doc[pno - 1]
            func[self.mode](
                page,
                textout,
                self.grid,
                self.fontsize,
                self.noformfeed,
                self.skip_empty,
                flags=flags,
            )
        textout.close()

# Usage
extractor = PDFExtractor("input.pdf", "output.txt")
extractor.extract_text()
class TextCleaner:
    @staticmethod
    def clean_text(text):
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(' +', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'\s+([.,;:])', r'\1', text)  # Remove space before punctuation
        return text.strip()


class Tokenizer:
    def __init__(self, encoding=EMBEDDING_ENCODING):
        self.tokenizer = tiktoken.get_encoding(encoding)

    def create_chunks(self, text):
        tokens = self.tokenizer.encode(text)
        i = 0
        #for i in tqdm(range(len(tokens)), desc="Tokenizing", unit="token"):

        while i < len(tokens):
            j = min(i + int(1.5 * TOKEN_CHUNK_SIZE), len(tokens))
            while j > i + int(0.5 * TOKEN_CHUNK_SIZE):
                chunk = self.tokenizer.decode(tokens[i:j])
                if chunk.endswith(".") or chunk.endswith("\n"):
                    break
                j -= 1
            if j == i + int(0.5 * TOKEN_CHUNK_SIZE):
                j = min(i + TOKEN_CHUNK_SIZE, len(tokens))
            yield tokens[i:j]
            i = j

class JSONFormatter:
    @staticmethod
    def format(data, author="Author Name", source="Source Information", timestamp="2023-01-01T00:00:00Z", tags=None):
        if tags is None:
            tags = ["tag1", "tag2"]
        
        formatted_data = {
            "data": [
                {
                    "id": f"unique_identifier_{index}",
                    "content": item,
                    "metadata": {
                        "author": author,
                        "source": source,
                        "timestamp": timestamp,
                        "tags": tags
                    }
                } for index, item in enumerate(data)
            ]
        }
        return json.dumps(formatted_data, indent=4)


class JSONProcessor:
    def __init__(self, file_path, output_path, text_cleaner, embedding_generator):
        self.file_path = file_path
        self.output_path = output_path
        self.text_cleaner = text_cleaner
        self.embedding_generator = embedding_generator

    def process_json_file(self):
        try:
            json_data = self._read_json_file()
            self._process_data(json_data)
            self._write_json_file(json_data)
        except Exception as e:
            print(f"Error processing JSON file: {str(e)}")
            sys.exit(1)

    def _read_json_file(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _process_data(self, json_data):
        if 'data' not in json_data or not isinstance(json_data['data'], list):
            raise ValueError("JSON file format is incorrect. Expected a 'data' field with an array.")

        batch_size = 120
        for i in tqdm(range(0, len(json_data['data']), batch_size), desc="Processing", unit="batch"):
            batch = [self.text_cleaner.clean_text(item['content']) for item in json_data['data'][i:i+batch_size]]
            embeddings_batch = self.embedding_generator.generate_embeddings_batch(batch)

            for index, embeddings in enumerate(embeddings_batch):
                json_data['data'][i + index]['embeddings'] = embeddings if embeddings else None

    def _write_json_file(self, json_data):
        with open(self.output_path, 'w', encoding='utf-8') as file:
            json.dump(json_data, file, indent=2, ensure_ascii=False)


class EmbeddingGenerator:
    def __init__(self, cohere_client):
        self.cohere_client = cohere_client

    def generate_embeddings_batch(self, texts):
        try:
            response = self.cohere_client.embed(
                model='embed-multilingual-v3.0',
                texts=texts,
                input_type='search_document'
            )
            return response.embeddings
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return [None] * len(texts)


def main(filepath):


    # Extract text from PDF
    extractor = PDFExtractor(filepath, "output.txt")
    text = extractor.extract_text()

    # Clean the extracted text
    cleaner = TextCleaner()
    cleaned_text = cleaner.clean_text(text)

    # Tokenize and chunk the cleaned text
    tokenizer = Tokenizer()
    print("Tokenizing and chunking text...")
    chunks = tokenizer.create_chunks(cleaned_text)
    text_chunks = [tokenizer.tokenizer.decode(chunk) for chunk in tqdm(chunks, desc="Processing", unit="chunk")]

    print("Formatting data as JSON...")
    json_data = JSONFormatter.format(text_chunks)  # Assuming JSONFormatter has a static method called 'format'

    intermediate_output_filename = os.path.splitext(os.path.basename(filepath))[0] + "_output.json"
    with open(intermediate_output_filename, 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)
    print(f"Intermediate data saved to {intermediate_output_filename}")


    # Add embeddings to JSON
    co_client = cohere.Client('MfkoPnOYfdjb9hln30HfWlvgcCYuf0TAeQ2zOx0g')  # Use your actual Cohere API key
    embedding_generator = EmbeddingGenerator(co_client)
    final_output_filename = os.path.splitext(os.path.basename(filepath))[0] + "_final_output.json"
    json_processor = JSONProcessor(intermediate_output_filename, final_output_filename, text_cleaner, embedding_generator)
    json_processor.process_json_file()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Information from PDF, Tokenize, and Add Embeddings")
    parser.add_argument("filepath", help="Path to the PDF file.")
    args = parser.parse_args()

    if args.filepath:
        main(args.filepath)
    else:
        print("Please specify the file path as an argument.")
        sys.exit(1)
