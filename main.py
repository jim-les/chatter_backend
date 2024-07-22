import nltk
from nltk.chat.util import Chat, reflections
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import yaml
from mku_pairs import pairs as mku_pairs
from nltk.corpus import wordnet


reflections = {
    "I am": "you are",
    "I was": "you were",
    "I": "you",
    "I'm": "you are",
    "I'd": "you would",
    "I've": "you have",
    "I'll": "you will",
    "my": "your",
    "you are": "I am",
    "you were": "I was",
    "your": "my",
    "yours": "mine",
    "you": "me",
    "me": "you"
}

# Create a ChatBot instance
chatbot = Chat(mku_pairs, reflections)


# Function to handle the chat interaction
def chat():
    print("Hi, I'm ChatBot! How can I help you today?")
    previous_response = None  # To track context in multi-turn conversation
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("ChatBot: Goodbye!")
            break
        
        # Tokenize user input
        tokens = word_tokenize(user_input)
        
        # Use NLTK's built-in methods for similarity and flexibility
        best_match_response = get_best_response(user_input)
        if best_match_response:
            print("ChatBot (best match):", best_match_response)
            continue
        
        # Multi-turn conversation handling
        if previous_response:
            response = chatbot.respond(previous_response + ' ' + user_input)
        else:
            response = chatbot.respond(user_input)
        
        print("ChatBot (from NLTK):", response)
        previous_response = response  # Update previous_response for context

def get_best_response(user_input):
    best_match = None
    best_match_similarity = 0.0
    
    for pattern, responses in mku_pairs:
        similarity = calculate_similarity(user_input.lower(), pattern.lower())
        if similarity > best_match_similarity:
            best_match_similarity = similarity
            best_match = responses[0]  # Select the first response for simplicity
    
    # You can define a threshold for similarity to decide if it's a good enough match
    if best_match_similarity >= 0.5:  # Adjust this threshold as needed
        return best_match
    else:
        return None
    
def generate_expected_response(user_input):
    # Implement logic to generate expected response based on user's query
    # Example: Generate response based on keywords in user_input
    if 'according to you' in user_input:
        return "According to my understanding, you might be asking..."
    elif 'request' in user_input:
        return "I was expecting you to ask..."
    else:
        return "I'm not sure I understand. Can you please clarify?"


def calculate_similarity(input_text, pattern):
    # Tokenize input_text and pattern
    tokens_input = set(word_tokenize(input_text.lower()))  # Convert to lowercase for case insensitivity
    tokens_pattern = set(word_tokenize(pattern.lower()))  # Convert to lowercase for case insensitivity
    
    # Find the intersection of tokens
    overlap = tokens_input.intersection(tokens_pattern)
    
    # Calculate simple token overlap similarity
    overlap_similarity = len(overlap) / (len(tokens_input) + len(tokens_pattern))
    
    # Use WordNet for more advanced similarity measure
    wordnet_similarity = calculate_wordnet_similarity(tokens_input, tokens_pattern)
    
    # Combine both similarities, giving more weight to WordNet similarity
    combined_similarity = 0.7 * wordnet_similarity + 0.3 * overlap_similarity
    
    return combined_similarity

def calculate_wordnet_similarity(tokens_input, tokens_pattern):
    max_similarity = 0.0
    
    # Calculate maximum similarity based on WordNet synsets
    for token1 in tokens_input:
        for token2 in tokens_pattern:
            synsets1 = wordnet.synsets(token1)
            synsets2 = wordnet.synsets(token2)
            
            # Calculate similarity between synsets using path_similarity
            for synset1 in synsets1:
                for synset2 in synsets2:
                    similarity = synset1.path_similarity(synset2)
                    if similarity is not None and similarity > max_similarity:
                        max_similarity = similarity
    
    return max_similarity if max_similarity is not None else 0.0

# Start chatting
if __name__ == "__main__":
    chat()