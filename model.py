import torch
from transformers import BertTokenizer, BertForSequenceClassification
import streamlit as st

@st.cache_resource
def get_model():
    """
    Loads and returns the BERT model and tokenizer.
    Uses Streamlit's caching to load the model only once.
    """
    try:
        # --- Use a fine-tuned model if available ---
        # Update this path to your fine-tuned model directory if you have one.
        # For example: model_name = './results/final_model'
        # tokenizer = BertTokenizer.from_pretrained(model_name)
        # model = BertForSequenceClassification.from_pretrained(model_name)
        
        # --- Fallback to a base pre-trained model ---
        # If a fine-tuned model is not found, we use a general-purpose base model.
        # Note: Its predictions will not be accurate for this specific task
        # without fine-tuning.
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        print("Loaded base 'bert-base-uncased' model. Fine-tuning is recommended for better accuracy.")

    except Exception as e:
        st.error(f"Error loading model: {e}")
        # As a last resort, if online models fail, create a dummy model for UI testing.
        # This part is unlikely to be triggered if there's an internet connection.
        from transformers import BertConfig
        config = BertConfig(num_labels=2)
        model = BertForSequenceClassification(config)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Tokenizer might still load
        st.warning("Could not load a pre-trained model. Using a dummy model for UI demonstration.")
        
    return tokenizer, model

def predict(text):
    """
    Takes a text string and returns the predicted probabilities for "Human" and "AI".
    """
    # Load the model and tokenizer
    tokenizer, model = get_model()

    # Prepare the input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Move tensors to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Get predictions
    with torch.no_grad():
        logits = model(**inputs).logits

    # Convert logits to probabilities
    # The model outputs two values (logits). We apply softmax to convert them to probabilities.
    # Logits for [Human, AI] -> Probabilities for [Human, AI]
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()

    # Let's assume the model's labels are: 0 -> Human, 1 -> AI
    # This is a convention we'd enforce during fine-tuning.
    prob_human = probabilities[0]
    prob_ai = probabilities[1]
    
    return {"Human": prob_human, "AI": prob_ai}
