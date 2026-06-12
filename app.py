import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os

st.title("🧠 Multi-Modal Semantic Image Search Engine")
st.write("Search through your images using natural language sentences powered by OpenAI's CLIP Model.")

# 1. Load the Pre-trained Multi-Modal AI Model from Hugging Face
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

# 2. Simulate a local image database folder
IMAGE_DIR = "image_database"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
    # Creating a placeholder blank image if folder is empty
    img = Image.new('RGB', (300, 300), color = (73, 109, 137))
    img.save(os.path.join(IMAGE_DIR, "placeholder_sample.jpg"))

# List all images in our database directory
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if image_files:
    st.sidebar.write(f"📁 Loaded {len(image_files)} images from `{IMAGE_DIR}/` folder.")
    
    # 3. User Text Query Input
    search_query = st.text_input("Enter what you want to find (e.g., 'a blue car', 'a landscape shot'):")
    
    if search_query:
        st.subheader("🔍 Search Results:")
        
        images = []
        for file in image_files:
            img_path = os.path.join(IMAGE_DIR, file)
            images.append(Image.open(img_path))
            
        # 4. Processing text query and images through CLIP to get matching vector scores
        inputs = processor(text=[search_query], images=images, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        
        # Calculate logits (similarity probabilities)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=0).detach().numpy()
        
        # Zip image names with their matching percentage scores
        results = sorted(zip(image_files, probs), key=lambda x: x[1][0], reverse=True)
        
        # Display images ranked by similarity matching score
        cols = st.columns(3)
        for idx, (img_name, score) in enumerate(results[:3]): # Display top 3 matches
            with cols[idx % 3]:
                img_path = os.path.join(IMAGE_DIR, img_name)
                st.image(Image.open(img_path), use_container_width=True)
                st.caption(f"Match Score: {score[0]*100:.2f}%")
else:
    st.warning("Please drop some JPG/PNG images into the 'image_database' folder to start searching!")
