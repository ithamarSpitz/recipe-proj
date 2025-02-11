# Initial imports and dataset loading
import pandas as pd, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
dataset = pd.read_csv('data/13k-recipes.csv', index_col=0)

# Data preprocessing:
# - Converts ingredients from list format to space-separated string
# - Fills empty values with blank strings
dataset['Ingredients'] = dataset['Ingredients'].apply(lambda x: " ".join(eval(x)))
dataset.fillna('', inplace=True)

# Hardware setup: Uses GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model initialization:
# - Uses Alibaba's multilingual language model
# - Creates tokenizer (text-to-numbers converter)
# - Loads model to GPU/CPU
model_name_or_path = 'Alibaba-NLP/gte-multilingual-base'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)

# Core function: Converts text to numerical embeddings
def get_text_embeddings(text):
    # Processes text: limits length to 512, adds padding, converts to tensor
    batch_dict = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    # Moves processed text to GPU/CPU
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
    # Generates embeddings using model
    outputs = model(**batch_dict)
    # Extracts specific embedding vector and converts to numpy
    embeddings = outputs.last_hidden_state[:, 0].detach().cpu().numpy()
    return embeddings

# Initialize storage for different types of embeddings
title_embeddings = []
ingredients_embeddings = []
instructions_embeddings = []
all_embeddings = []

# Main processing loop:
# - Processes each recipe in dataset
# - Creates embeddings for title, ingredients, instructions
# - Creates embedding for complete recipe text
for i, row in dataset.iterrows():
    title_embeddings.append(get_text_embeddings(row['Title']))
    ingredients_embeddings.append(get_text_embeddings(row['Ingredients']))
    instructions_embeddings.append(get_text_embeddings(row['Instructions']))
    all_embeddings.append(get_text_embeddings(row['Title'] + ' ' + row['Ingredients'] + ' ' + row['Instructions']))

# Combines individual embeddings into unified arrays
title_embeddings = np.concatenate(title_embeddings, axis=0)
ingredients_embeddings = np.concatenate(ingredients_embeddings, axis=0)
instructions_embeddings = np.concatenate(instructions_embeddings, axis=0)
all_embeddings = np.concatenate(all_embeddings, axis=0)

# Saves all embeddings to files for later use
np.save('data/title_embeddings.npy', title_embeddings)
np.save('data/ingredients_embeddings.npy', ingredients_embeddings)
np.save('data/instructions_embeddings.npy', instructions_embeddings)
np.save('data/all_embeddings.npy', all_embeddings)
