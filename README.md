# Multi-Modal Semantic Image Search Engine

A next-generation image retrieval application that allows users to search through an unlabelled image database using descriptive, natural language queries instead of traditional keyword metadata.

## Technical Stack
* **AI Model:** OpenAI's CLIP (via Hugging Face Transformers)
* **Frontend UI:** Streamlit
* **Core Libraries:** PyTorch, Pillow (PIL), Python

## Key Architecture Concepts
* **Multi-Modal Embeddings:** Leverages a shared neural network space where text sentences and raw image pixel matrices are mapped directly to corresponding high-dimensional vector coordinates.
* **Zero-Shot Evaluation:** Utilizes Contrastive Language-Image Pre-training to recognize complex image contexts natively without requiring localized training or fine-tuning datasets.

## How to Run Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/Praveen-dev18/semantic-image-search.git](https://github.com/Praveen-dev18/semantic-image-search.git)
2. Install the necessary machine learning and web UI dependencies:
   pip install streamlit transformers torch pillow
3. Create a folder named image_database in the root directory, drop your images inside it, and boot up the local web engine:
   streamlit run app.py
