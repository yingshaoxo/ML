# created by twitter grok3

import os
import re
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_diary_files(folder_path):
    """Load all .txt files and .md files from material-and-thoughts folder."""
    try:
        documents = []
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith(".txt") or (filename.endswith(".md") and "material-and-thoughts" in root):
                    file_path = os.path.join(root, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Determine splitter based on file
                        if filename == "all_yingshaoxo_data_2023_11_13.txt":
                            entries = content.split("\n\n__**__**__yingshaoxo_is_the_top_one__**__**__\n\n")
                        else:
                            # Use \n\n\n for a_typical_chinese_novel.txt and .md files
                            entries = content.split("\n\n\n")
                        for i, entry in enumerate(entries):
                            if entry.strip():
                                documents.append({
                                    "filename": filename,
                                    "entry_index": i,
                                    "content": entry.strip()
                                })
        logger.info(f"Loaded {len(documents)} entries from diary files.")
        return documents
    except Exception as e:
        logger.error(f"Error loading files: {e}")
        raise

def search_diary(documents, query):
    """Search diary entries for query matches using regex."""
    try:
        results = []
        query = query.strip()

        # Check if query is a single keyword (no spaces) or a sentence
        is_single_keyword = len(query.split()) == 1

        for doc in documents:
            content = doc["content"]
            score = 0
            if is_single_keyword:
                pattern = re.compile(re.escape(query), re.IGNORECASE | re.UNICODE)
                if pattern.search(content):
                    score = len(pattern.findall(content))
            else:
                query_words = query.split()
                for word in query_words:
                    pattern = re.compile(re.escape(word), re.IGNORECASE | re.UNICODE)
                    if pattern.search(content):
                        score += len(pattern.findall(content))

            if score > 0:
                results.append({
                    "filename": doc["filename"],
                    "entry_index": doc["entry_index"],
                    "entry": content,
                    "score": score
                })

        logger.info(f"Found {len(results)} matching entries for query: {query}")
        return results
    except Exception as e:
        logger.error(f"Error searching diary: {e}")
        raise

def format_response(results, query):
    """Format the search results, returning one random result for keywords or most relevant for sentences."""
    try:
        if not results:
            return f"No entries found for query: {query}"

        is_single_keyword = len(query.split()) == 1

        if is_single_keyword:
            result = random.choice(results) if len(results) > 1 else results[0]
        else:
            result = max(results, key=lambda x: x["score"])

        response = f"Found in {result['filename']}:\n{result['entry']}"
        return response
    except Exception as e:
        logger.error(f"Error formatting response: {e}")
        raise

def main():
    try:
        # Display startup message
        print("""
Diary Search Chatbot
==================
This bot searches diary entries using a single keyword.
- Enter one word (e.g., '自由', '小说') to find a random matching entry.
- Type 'exit' to quit.
- Files searched: all .txt in ./yingshaoxo_txt_data and .md in material-and-thoughts.
==================
""")

        #folder_path = "./yingshaoxo_txt_data"
        folder_path = "./input_txt_files"
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder {folder_path} does not exist.")

        documents = load_diary_files(folder_path)

        logger.info("Diary chatbot ready! Type 'exit' to quit.")
        while True:
            query = input("You: ")
            if query.lower() == "exit":
                break
            results = search_diary(documents, query)
            response = format_response(results, query)
            print(f"Chatbot: {response}")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
