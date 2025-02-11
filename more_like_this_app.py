# Basic imports - libraries needed for the application
import tkinter as tk  # Main library for creating GUI applications in Python
from tkinter import ttk, scrolledtext  # ttk provides themed widgets, scrolledtext for scrollable text areas
import numpy as np  # For numerical operations and array handling
from transformers import AutoTokenizer, AutoModel  # For natural language processing
import pandas as pd  # For data manipulation and reading CSV files

class RecipeSimilarityApp:
    def __init__(self, root):
        # Initialize the main window
        self.root = root
        self.root.title("Recipe Similarity Search")  # Set window title

        # Load the language model for text processing
        model_name_or_path = 'Alibaba-NLP/gte-multilingual-base'  # Specify which model to use
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)  # Tool to convert text into numbers
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)  # The actual AI model

        # Load pre-computed embeddings (numerical representations) of recipes
        self.all_embeddings = np.load('data/all_embeddings.npy')  # Combined embeddings
        self.title_embeddings = np.load('data/title_embeddings.npy')  # Title-only embeddings
        self.ingredients_embeddings = np.load('data/ingredients_embeddings.npy')  # Ingredients-only embeddings
        self.instructions_embeddings = np.load('data/instructions_embeddings.npy')  # Instructions-only embeddings

        # Load recipe data from CSV file
        self.recipes = pd.read_csv('data/13k-recipes.csv', index_col=0).fillna("").to_dict(orient='records')

        # Create the search frame (top part of the window)
        search_frame = ttk.Frame(root, padding="10")
        search_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))  # Position frame with expansion in all directions
        root.grid_columnconfigure(0, weight=1)  # Allow horizontal expansion
        root.grid_rowconfigure(1, weight=1)  # Allow vertical expansion

        # Create dropdown menu for search type selection
        self.search_type = ttk.Combobox(search_frame, values=[
            "All", "Title Only", "Ingredients Only", "Instructions Only"
        ])
        self.search_type.set("All")  # Set default value
        self.search_type.grid(row=0, column=0, padx=5)  # Position dropdown

        # Create search input field
        self.search_entry = ttk.Entry(search_frame, width=50)  # Text input box
        self.search_entry.grid(row=0, column=1, padx=5)  # Position input box

        # Create search button
        ttk.Button(search_frame, text="Search", command=self.search).grid(row=0, column=2)

        # Create results frame (bottom part of the window)
        results_frame = ttk.Frame(root, padding="10")
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_rowconfigure(0, weight=1)

        # Create scrollable text area for results
        self.results_text = scrolledtext.ScrolledText(results_frame, height=20, font=('Arial', 11))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure text styles for different parts of the results
        self.results_text.tag_configure('title', font=('Arial', 12, 'bold'))  # Style for recipe titles
        self.results_text.tag_configure('header', font=('Arial', 11, 'bold'))  # Style for section headers
        self.results_text.tag_configure('similarity', font=('Arial', 10, 'italic'))  # Style for similarity scores
        self.results_text.tag_configure('search_type', font=('Arial', 12, 'bold'), foreground='navy')  # Style for search type header

        # Create frame for additional buttons
        self.buttons_frame = ttk.Frame(results_frame)
        self.buttons_frame.grid(row=1, column=0, pady=10)

        self.current_query_embedding = None  # Store the current search query's numerical representation

    def get_current_embeddings(self):
        # Select which embeddings to use based on search type
        search_type = self.search_type.get()
        if search_type == "Title Only":
            return self.title_embeddings
        elif search_type == "Ingredients Only":
            return self.ingredients_embeddings
        elif search_type == "Instructions Only":
            return self.instructions_embeddings
        return self.all_embeddings

    def cosine_similarity(self, vec1, vec2):
        # Calculate similarity between two vectors (mathematical comparison of text representations)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def find_similar(self, query_embedding):
        # Find the 5 most similar recipes to the search query
        current_embeddings = self.get_current_embeddings()
        similarities = [
            (i, self.cosine_similarity(query_embedding, recipe_embedding))
            for i, recipe_embedding in enumerate(current_embeddings)
        ]
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:5]

    def search(self):
        # Process the search query and find similar recipes
        query = self.search_entry.get()  # Get search text
        # Convert text to numbers the model can understand
        batch_dict = self.tokenizer(query, max_length=512, padding=True, truncation=True, return_tensors='pt')
        # Get numerical representation of the query
        outputs = self.model(**batch_dict)
        self.current_query_embedding = outputs.last_hidden_state[:, 0].detach().cpu().numpy().flatten()
        # Find similar recipes
        similar_recipes = self.find_similar(self.current_query_embedding)
        # Display results
        self.display_results(similar_recipes)

    def display_results(self, similar_recipes):
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        for widget in self.buttons_frame.winfo_children():
            widget.destroy()

        # Show search type
        search_type = self.search_type.get()
        self.results_text.insert(tk.END, f"Search type: {search_type}\n\n", 'search_type')

        # Display each similar recipe with formatting
        for i, (idx, similarity) in enumerate(similar_recipes):
            recipe = self.recipes[idx]

            # Add recipe title
            self.results_text.insert(tk.END, f"{i + 1}. {recipe['Title']}\n", 'title')

            # Add ingredients section
            self.results_text.insert(tk.END, "Ingredients:\n", 'header')
            self.results_text.insert(tk.END, f"{recipe['Ingredients']}\n\n")

            # Add instructions section
            self.results_text.insert(tk.END, "Instructions:\n", 'header')
            self.results_text.insert(tk.END, f"{recipe['Instructions'][:200]}...\n\n")

            # Add similarity score
            self.results_text.insert(tk.END, f"Similarity: {similarity:.2f}\n\n", 'similarity')

# Start the application when the script is run
if __name__ == "__main__":
    root = tk.Tk()  # Create main window
    app = RecipeSimilarityApp(root)  # Create app instance
    root.mainloop()  # Start the application loop
