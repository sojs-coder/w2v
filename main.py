import numpy as np
from collections import defaultdict, Counter
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class CBOW:
    """
    A simple implementation of the Continuous Bag of Words (CBOW) architecture
    for learning word embeddings.
    """
    
    def __init__(self, text, window_size=2, embedding_dim=100, epochs=25, learning_rate=0.01):
        """
        Initialize the CBOW model.
        
        Args:
            text (list): List of sentences, where each sentence is a list of words
            window_size (int): Size of context window on each side of target word
            embedding_dim (int): Dimension of word embeddings
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate for gradient descent
        """
        self.text = text
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        # Process the text to create vocabulary and mappings
        self._process_text()
        
        # Initialize model weights (embedding matrices)
        self._initialize_weights()
        
    def _process_text(self):
        """
        Process the text to create vocabulary and mappings between words and indices.
        """
        # Flatten list of sentences to get all words
        all_words = [word for sentence in self.text for word in sentence]
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Create vocabulary (sorted by frequency, then alphabetically for ties)
        self.vocab = sorted(word_counts.keys(), key=lambda w: (-word_counts[w], w))
        
        # Create mapping from words to indices and vice versa
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        # Vocabulary size
        self.vocab_size = len(self.vocab)
        
        print(f"Vocabulary size: {self.vocab_size}")
        
    def _initialize_weights(self):
        """
        Initialize the weights for the input->hidden and hidden->output layers.
        """
        # Input -> Hidden weights (embedding matrix)
        # Each row represents a word embedding
        self.W1 = np.random.uniform(-0.8, 0.8, (self.vocab_size, self.embedding_dim))
        
        # Hidden -> Output weights (context matrix)
        # Each column represents a word context
        self.W2 = np.random.uniform(-0.8, 0.8, (self.embedding_dim, self.vocab_size))
        
    def _generate_training_data(self):
        """
        Generate training data pairs of (context_words, target_word).
        
        Returns:
            list: List of tuples (context_word_indices, target_word_index)
        """
        training_data = []
        
        # Process each sentence
        for sentence in self.text:
            indices = [self.word2idx[word] for word in sentence if word in self.word2idx]
            
            # For each word in the sentence
            for center_idx, target_idx in enumerate(indices):
                # Get context indices within window size
                context_indices = []
                for i in range(max(0, center_idx - self.window_size), 
                               min(len(indices), center_idx + self.window_size + 1)):
                    if i != center_idx:  # Skip the target word
                        context_indices.append(indices[i])
                
                # Only keep if we have context words
                if context_indices:
                    training_data.append((context_indices, target_idx))
        
        return training_data
    
    def _forward_pass(self, context_indices):
        """
        Perform a forward pass through the network.
        
        Args:
            context_indices (list): Indices of context words
            
        Returns:
            tuple: (hidden layer output, output probabilities)
        """
        # Get embeddings for all context words
        context_embeddings = self.W1[context_indices]
        
        # Average the context embeddings (this is the 'continuous bag')
        hidden = np.mean(context_embeddings, axis=0)
        
        # Project to output layer
        output = np.dot(hidden, self.W2)
        
        # Apply softmax to get probabilities
        exp_output = np.exp(output - np.max(output))  # Subtract max for numerical stability
        softmax_output = exp_output / np.sum(exp_output)
        
        return hidden, softmax_output
    
    def _backward_pass(self, hidden, output, context_indices, target_idx):
        """
        Perform a backward pass to update weights.
        
        Args:
            hidden (numpy.array): Hidden layer output
            output (numpy.array): Output probabilities
            context_indices (list): Indices of context words
            target_idx (int): Index of target word
        """
        # Calculate output layer error
        # Create one-hot encoded target
        target = np.zeros(self.vocab_size)
        target[target_idx] = 1
        
        # Calculate output error
        output_error = output - target
        
        # Update hidden->output weights (W2)
        gradient_W2 = np.outer(hidden, output_error)
        self.W2 -= self.learning_rate * gradient_W2
        
        # Propagate error to hidden layer
        hidden_error = np.dot(self.W2, output_error)
        
        # Update input->hidden weights (W1) for each context word
        for idx in context_indices:
            # The error is distributed equally among all context words
            self.W1[idx] -= self.learning_rate * hidden_error / len(context_indices)
    
    def train(self):
        """
        Train the CBOW model using stochastic gradient descent.
        """
        # Generate training data
        training_data = self._generate_training_data()
        print(f"Training on {len(training_data)} context-target pairs")
        
        # Training loop
        losses = []
        for epoch in range(self.epochs):
            # Shuffle training data for each epoch
            random.shuffle(training_data)
            
            total_loss = 0
            # Process each training example
            for context_indices, target_idx in training_data:
                # Forward pass
                hidden, output = self._forward_pass(context_indices)
                
                # Calculate loss (cross-entropy loss)
                loss = -np.log(output[target_idx] + 1e-10)  # Add small value to avoid log(0)
                total_loss += loss
                
                # Backward pass
                self._backward_pass(hidden, output, context_indices, target_idx)
            
            # Average loss for this epoch
            avg_loss = total_loss / len(training_data)
            losses.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        # Plot the learning curve
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.epochs + 1), losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('CBOW Training Loss')
        plt.grid(True)
        plt.show()
        
        return losses
    
    def get_word_embedding(self, word):
        """
        Get the embedding vector for a specific word.
        
        Args:
            word (str): The word to get the embedding for
            
        Returns:
            numpy.array: The embedding vector
        """
        if word in self.word2idx:
            return self.W1[self.word2idx[word]]
        else:
            print(f"Word '{word}' not in vocabulary")
            return None
    
    def get_most_similar(self, word, n=5):
        """
        Find the n most similar words to the given word.
        
        Args:
            word (str): The word to find similar words for
            n (int): Number of similar words to return
            
        Returns:
            list: List of (word, similarity) tuples
        """
        if word not in self.word2idx:
            print(f"Word '{word}' not in vocabulary")
            return []
        
        # Get the embedding for the given word
        word_embedding = self.get_word_embedding(word)
        
        # Calculate cosine similarity with all other words
        similarities = []
        for idx, w in self.idx2word.items():
            if w == word:
                continue  # Skip the input word
                
            other_embedding = self.W1[idx]
            
            # Compute cosine similarity
            similarity = np.dot(word_embedding, other_embedding) / (
                np.linalg.norm(word_embedding) * np.linalg.norm(other_embedding)
            )
            similarities.append((w, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n]
    
    def visualize_embeddings(self, words=None, n=50):
        """
        Visualize word embeddings in 2D using PCA.
        
        Args:
            words (list, optional): Specific words to visualize. If None, uses top n frequent words.
            n (int): Number of words to visualize if words is None
        """
        # If no specific words are provided, use the most frequent words
        if words is None:
            all_words = [word for sentence in self.text for word in sentence]
            word_counts = Counter(all_words)
            words = [word for word, _ in word_counts.most_common(n)]
        
        # Filter words that are in vocabulary
        words = [word for word in words if word in self.word2idx]
        
        if not words:
            print("No valid words to visualize")
            return
        
        # Get embeddings for these words
        word_embeddings = [self.W1[self.word2idx[word]] for word in words]
        
        # Apply PCA to reduce dimensionality to 2
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(word_embeddings)
        
        # Create a scatter plot
        plt.figure(figsize=(15, 10))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        
        # Add word labels
        for i, word in enumerate(words):
            plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        fontsize=9, alpha=0.8)
        
        plt.title("Word Embeddings Visualization")
        plt.grid(True)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Sample text corpus
    corpus = [
        "the quick brown fox jumps over the lazy dog".split(),
        "the dog barks at the fox".split(),
        "quick brown foxes leap over lazy dogs".split(),
        "dogs and foxes are animals".split(),
        "the quick rabbit jumps over the fence".split(),
        "the turtle is slow and steady".split(),
        "slow and steady wins the race".split(),
        "the hare is quick but lazy".split(),
        "the fox is red and cunning".split(),
        "the dog is friendly and loyal".split(),
        "cats are independent animals".split(),
        "birds fly in the sky".split(),
        "fish swim in the water".split(),
        "the sky is blue".split(),
        "the water is clear".split(),
        "the grass is green".split(),
        "the sun is bright and warm".split(),
        "the moon shines at night".split(),
        "stars twinkle in the night sky".split(),
        "the earth orbits around the sun".split(),
    ]
    
    # Create and train CBOW model
    cbow_model = CBOW(
        text=corpus,
        window_size=2,
        embedding_dim=50,
        epochs=50,
        learning_rate=0.01
    )
    
    # Train the model
    losses = cbow_model.train()
    
    # Get similar words examples
    print("\nSimilar words to 'fox':")
    similar_fox = cbow_model.get_most_similar('fox', n=5)
    for word, similarity in similar_fox:
        print(f"{word}: {similarity:.4f}")
    
    print("\nSimilar words to 'quick':")
    similar_quick = cbow_model.get_most_similar('quick', n=5)
    for word, similarity in similar_quick:
        print(f"{word}: {similarity:.4f}")
    
    print("\nSimilar words to 'dog':")
    similar_dog = cbow_model.get_most_similar('dog', n=5)
    for word, similarity in similar_dog:
        print(f"{word}: {similarity:.4f}")
    
    # Visualize word embeddings
    cbow_model.visualize_embeddings(n=30)