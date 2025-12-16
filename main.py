# main.py
#
# This script provides a template for fine-tuning a BERT model to classify
# text as either human-written or machine-generated.
#
# Installation:
# pip install transformers torch scikit-learn pandas
#
# Usage:
# 1. Prepare your dataset as a CSV file with 'text' and 'label' columns.
#    - label 0: Human-written
#    - label 1: Machine-generated
# 2. Update the `file_path` variable in the `train()` function.
# 3. Run the script: python main.py

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

class TextDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for text classification."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Create a dictionary for the item, ensuring all values are tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Add the label for the item
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        # Return the total number of items in the dataset
        return len(self.labels)

def train():
    """
    This function handles the model training process.
    """
    # --- 1. Load and Prepare Data ---
    # IMPORTANT: Replace this with the actual path to your dataset.
    file_path = 'path/to/your/dataset.csv'
    try:
        # Load data from a CSV file into a pandas DataFrame.
        # The CSV should have two columns: 'text' for the content and 'label' for the classification (0 or 1).
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{file_path}'.")
        print("Please create a dummy CSV file with 'text' and 'label' columns or update the path.")
        # Create a dummy DataFrame for demonstration purposes if the file doesn't exist
        df = pd.DataFrame({
            'text': ["This is a human-written sentence.", "This is a machine-generated sentence."] * 10,
            'label': [0, 1] * 10
        })
        print("Using a dummy dataset for demonstration.")

    # Split the dataset into training and validation sets (80% train, 20% validation).
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    # --- 2. Tokenize Text ---
    # Load the tokenizer for the 'bert-base-uncased' model.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize the training and validation texts.
    # `truncation=True` ensures that sequences longer than the model's max length are cut short.
    # `padding=True` adds padding to shorter sequences to make them all the same length.
    # `max_length=512` sets the maximum sequence length for BERT.
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)

    # --- 3. Create PyTorch Datasets ---
    # Create dataset objects for the training and validation sets.
    train_dataset = TextDataset(train_encodings, list(train_labels))
    val_dataset = TextDataset(val_encodings, list(val_labels))

    # --- 4. Configure and Train the Model ---
    # Load the pre-trained BERT model for sequence classification with 2 labels (human/machine).
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Define the training arguments.
    training_args = TrainingArguments(
        output_dir='./results',              # Directory to save model checkpoints and logs.
        num_train_epochs=3,                  # Total number of training epochs.
        per_device_train_batch_size=8,       # Batch size for training.
        per_device_eval_batch_size=16,       # Batch size for evaluation.
        warmup_steps=500,                    # Number of steps for the learning rate to warm up.
        weight_decay=0.01,                   # Strength of weight decay for regularization.
        logging_dir='./logs',                # Directory for storing logs.
        logging_steps=10,                    # Log metrics every 10 steps.
        evaluation_strategy="epoch",         # Run evaluation at the end of each epoch.
        save_strategy="epoch",               # Save a model checkpoint at the end of each epoch.
        load_best_model_at_end=True,         # Load the best model found during training at the end.
    )

    # Initialize the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Start the training process.
    print("Starting model training...")
    trainer.train()
    print("Training finished.")

    # Save the final model and tokenizer
    final_model_path = './results/final_model'
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final model saved to {final_model_path}")


def predict(text_to_classify, model_path='./results/final_model'):
    """
    This function uses the fine-tuned model to make a prediction on new text.
    """
    try:
        # Load the fine-tuned model and tokenizer from the specified path.
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
    except OSError:
        print(f"Error: Model not found at '{model_path}'.")
        print("Please train the model first by running the `train()` function.")
        return

    # Prepare the input text for the model.
    # `return_tensors="pt"` specifies that the output should be PyTorch tensors.
    inputs = tokenizer(text_to_classify, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Move tensors to the same device as the model (GPU if available, otherwise CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Make a prediction. `torch.no_grad()` disables gradient calculations to save memory and speed up inference.
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted class ID by finding the index of the highest logit.
    predicted_class_id = logits.argmax().item()

    # Map the class ID to the human-readable label.
    # (Assuming 0 for Human-written, 1 for Machine-generated as defined during training)
    prediction = "Machine-generated" if predicted_class_id == 1 else "Human-written"

    print(f"\n--- Prediction ---")
    print(f"Text: '{text_to_classify}'")
    print(f"Predicted as: {prediction}")
    print(f"------------------")


if __name__ == "__main__":
    # This block runs when the script is executed directly.
    
    # Step 1: Train the model.
    # This will load data, fine-tune the BERT model, and save the result in the './results' directory.
    train()

    # Step 2: Use the trained model for inference.
    # This loads the fine-tuned model from the './results/final_model' directory and classifies a new paragraph.
    new_paragraph_to_test = "The quick brown fox jumps over the lazy dog. This sentence was likely written by a person."
    predict(new_paragraph_to_test)
