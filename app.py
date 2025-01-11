import pandas as pd
import openai
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
import faiss
import streamlit as st
from datetime import datetime

# Set OpenAI API Key as the environment variable
_ = load_dotenv(find_dotenv())  # Load environment variables from .env file
openai.api_key = os.getenv('OPENAI_API_KEY')

# Path to your insurance CSV file
INSURANCE_CSV_PATH = r"C:\Users\ajayk\OneDrive\Desktop\Team11\chatbot sample\Input\insurance.csv"  # Update with your actual file path

# Function to load insurance data
def load_insurance_data():
    df = pd.read_csv(INSURANCE_CSV_PATH)
    
    # Check for missing values in 'question' or 'answer' columns and remove them
    df = df.dropna(subset=['question', 'answer'])
    
    return df

# Function to compute embeddings for a list of texts using OpenAI
def compute_embeddings(texts):
    response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
    embeddings = response['data']
    return [embedding['embedding'] for embedding in embeddings]

# Function to create and index embeddings using Faiss
def create_faiss_index(embeddings):
    dimension = len(embeddings[0])  # Embedding dimension (should match the model output)
    index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance-based index
    index.add(np.array(embeddings).astype(np.float32))  # Add embeddings to the index
    return index

# Function to perform similarity search based on the user query
def get_similar_entries(prompt, index, master_df, k=1):
    query_embedding = compute_embeddings([prompt])[0]  # Compute the query embedding
    query_embedding = np.array(query_embedding).astype(np.float32)  # Convert to correct type

    # Perform the similarity search
    distances, indices = index.search(np.array([query_embedding]), k)
    
    # Ensure indices are valid and fetch the top-k results
    if len(indices[0]) > 0:
        top_k_results = master_df.iloc[indices[0]].reset_index(drop=True)
        return top_k_results, distances[0]
    else:
        # Return empty DataFrame if no valid result is found
        return pd.DataFrame(), []

# Define a function to handle small talk or predefined responses (no greetings)
def handle_small_talk(prompt):
    farewells = ["bye", "goodbye", "see you", "take care"]
    
    # Convert user input to lowercase for easier matching
    prompt = prompt.lower()
    
    if any(farewell in prompt for farewell in farewells):
        return "Goodbye! Have a great day!"
    else:
        return None  # Return None if no small talk match found

# Streamlit app
def main():
    st.title("Your InsureAssist! 🛡️🚑🏥")

    # Load insurance data and compute embeddings
    master_df = load_insurance_data()
    if 'embeddings' not in st.session_state:
        # Compute embeddings for the entire dataset (questions column)
        st.session_state.embeddings = compute_embeddings(master_df['question'].tolist())
        st.session_state.index = create_faiss_index(st.session_state.embeddings)  # Create the FAISS index
    
    # Initialize session state for messages if not already
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Sidebar for new chat and toggling chat history visibility
    with st.sidebar:
        st.button('Start new chat', on_click=reset_conversation)

        # Toggle for showing/hiding chat history
        if 'show_history' not in st.session_state:
            st.session_state.show_history = False  # Default to not showing history
        
        if st.button('Show Chat History'):
            st.session_state.show_history = not st.session_state.show_history

    # Display current chat messages if chat history is visible
    if 'messages' in st.session_state and len(st.session_state.messages) > 0:
        if st.session_state.show_history:
            for message in st.session_state.messages:
                if message["role"] != "system":
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Welcome to InsureAssist! How can I help you?"):
        # Debugging: Print the received prompt
        print(f"User input: {prompt}")
        
        # Check if the query is small talk
        response_text = handle_small_talk(prompt)
        
        if response_text is None:
            # Perform similarity search if no small talk detected
            relevant_entries, distances = get_similar_entries(prompt, st.session_state.index, master_df, k=1)  # Get the top 1 most similar entry
            
            # Debugging: Print results of the similarity search
            print(f"Relevant entries: {relevant_entries}")
            print(f"Distances: {distances}")
            
            # Check if we have valid results
            if relevant_entries.empty:
                response_text = "Sorry, I couldn't find an answer to that. Could you please rephrase your question?"
            else:
                # Fetch the exact answer from the 'answer' column
                response_text = relevant_entries['answer'].values[0]
        
        # Display the user's input and the assistant's response
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)

# Function to reset conversation
def reset_conversation():
    st.session_state.messages = []

if __name__ == "__main__":
    main()
