import streamlit as st
from PIL import Image
import numpy as np
import os
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

# Initialize Chroma DB client and embedding function
db_path = r"db"  # Add your db path here
client = chromadb.PersistentClient(path=db_path)
embedding_function = OpenCLIPEmbeddingFunction()

collection = client.get_or_create_collection(
    name='multimodal_collection3',
    embedding_function=embedding_function
)

# Function to get image embeddings and query related images
def get_image_embeddings(image):
    img = Image.open(image)
    img_ar = np.array(img)
    emb = embedding_function._encode_image(img_ar)
    
    docs = collection.query(
        query_embeddings=[emb],
        n_results=5
    )
    print(docs)
    
    return docs['ids'][0], docs['distances'][0],docs["metadatas"][0]

# Streamlit interface
st.title("Image Search with ChromaDB")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Searching for related images...")

    # Get related images
    related_ids, distances,metadata = get_image_embeddings(uploaded_file)

    # Display related images
    for i, img_id in enumerate(related_ids):
        img_path = os.path.join("image_data", img_id)  # Adjust the path if necessary
        if os.path.exists(img_path):
            st.image(img_path, caption=f"Related Image (Name:{img_id}  Distance: {distances[i]:.4f})", use_column_width=True)
        else:
            st.write(f"Image {img_id} not found in the directory.")
