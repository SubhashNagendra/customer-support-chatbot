import pandas as pd
import openai
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
import pickle
import faiss

# Set the OPENAI API Key as the environment variable
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')

# Config Paths
INPUT_FILE_NAME = "insurance.csv"
EMBEDDING_FILE_NAME = "embedding_array.pickle"
INPUT_FILE_DIR = "input"
OUTPUT_FILE_DIR = "output_test"
OUTPUT_MASTERDATA_FILE_NAME = "insurance_masterdata.pickle"
OUTPUT_INDEX_FILE_NAME = "index.pickle"


def create_text(row):
    # Process the text before sending it to LLM
    return f"question- {row['question']}\nanswer- {row['answer']}"


def generate_embedding_array(embeddings, embedding_file_name, output_file_dir):
    # Generate embeddings in numpy array and save it in a pickle format
    all_embeddings = [i['embedding'] for i in embeddings]
    embedding_array = np.array(all_embeddings)

    with open(os.path.join(output_file_dir, embedding_file_name), 'wb') as pickle_file:
        pickle.dump(embedding_array, pickle_file)


def create_faiss_index(embeddings_path, output_index_file_name, output_file_dir):
    # Define the dimensions for OpenAI embedding model which is 1536
    d = 1536
    index = faiss.IndexFlatIP(d)

    # Open the embedding saved in numpy array
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f).astype(np.float32)

    # Add embedding to the index
    index.add(embeddings)

    # Save the index pickle file to the output directory
    with open(os.path.join(output_file_dir, output_index_file_name), 'wb') as file:
        pickle.dump(index, file)


def convert_masterdata_to_pickle(df, output_masterdata_file_name, output_file_dir):
    # Save the DataFrame as a pickle file
    df.to_pickle(os.path.join(output_file_dir, output_masterdata_file_name))


def main():
    try:
        # Create the input file path
        root_dir = os.path.dirname(os.path.abspath(__file__))
        input_file_path = os.path.join(root_dir, INPUT_FILE_DIR, INPUT_FILE_NAME)

        # Read the master data CSV file
        df = pd.read_csv(input_file_path)

        # Create the input for embedding creation
        df['text'] = df.apply(create_text, axis=1)

        # Generate embeddings for all the rows
        response = openai.Embedding.create(
            input=df['text'].tolist(),
            model="text-embedding-ada-002"
        )
        embeddings = response['data']

        # Create numpy array for embeddings
        generate_embedding_array(embeddings, EMBEDDING_FILE_NAME, OUTPUT_FILE_DIR)

        # Create FAISS index
        create_faiss_index(os.path.join(root_dir, OUTPUT_FILE_DIR, EMBEDDING_FILE_NAME), OUTPUT_INDEX_FILE_NAME, OUTPUT_FILE_DIR)

        # Create master data for input file
        convert_masterdata_to_pickle(df, OUTPUT_MASTERDATA_FILE_NAME, OUTPUT_FILE_DIR)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
