from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from nltk.chat.util import Chat
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from mku_pairs import pairs as mku_pairs
import nltk
from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext

# Initialize NLTK resources if needed
nltk.download('punkt')
nltk.download('wordnet')

app = FastAPI()

# All origins
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Mock database
users_db = {}

# Reflections for chatbot responses
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

# Create ChatBot instance
chatbot = Chat(mku_pairs, reflections)

# Define request body models
class ChatInput(BaseModel):
    message: str

class User(BaseModel):
    username: str
    password: str  # 4-digit PIN as string

class UserInDB(BaseModel):
    username: str
    hashed_password: str

# Function to handle chat interaction
def get_bot_response(user_input):
    tokens = word_tokenize(user_input)
    best_match_response = get_best_response(user_input)
    if best_match_response:
        return best_match_response
    return chatbot.respond(user_input)

# Function to find best response based on similarity
def get_best_response(user_input):
    best_match = None
    best_match_similarity = 0.0
    
    for pattern, responses in mku_pairs:
        similarity = calculate_similarity(user_input.lower(), pattern.lower())
        if similarity > best_match_similarity:
            best_match_similarity = similarity
            best_match = responses[0]  # Select the first response for simplicity
    
    # Define a threshold for similarity
    if best_match_similarity >= 0.5:
        return best_match
    else:
        return None

def calculate_similarity(user_input, pattern):
    tokens_input = word_tokenize(user_input)
    tokens_pattern = word_tokenize(pattern)
    
    wordnet_similarity = calculate_wordnet_similarity(tokens_input, tokens_pattern)
    
    return wordnet_similarity

# Calculate WordNet similarity
def calculate_wordnet_similarity(tokens_input, tokens_pattern):
    max_similarity = 0.0
    
    for token1 in tokens_input:
        for token2 in tokens_pattern:
            synsets1 = wordnet.synsets(token1)
            synsets2 = wordnet.synsets(token2)
            
            for synset1 in synsets1:
                for synset2 in synsets2:
                    similarity = synset1.path_similarity(synset2)
                    if similarity is not None and similarity > max_similarity:
                        max_similarity = similarity
    
    return max_similarity if max_similarity is not None else 0.0

@app.get("/")
def read_root():
    return {"message": "Welcome to the ChatBot API!"}


@app.post("/chat/")
def chat_endpoint(chat_input: ChatInput):
    user_input = chat_input.message
    bot_response = get_bot_response(user_input)
    return {"message": bot_response}


# Utility functions for user management
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


@app.post("/signup/")
async def signup(user: User):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    if len(user.password) != 4 or not user.password.isdigit():
        raise HTTPException(status_code=400, detail="Password must be a 4-digit PIN")
    hashed_password = pwd_context.hash(user.password)
    users_db[user.username] = UserInDB(username=user.username, hashed_password=hashed_password)
    return {"message": "User created successfully"}


# Login endpoint
@app.post("/login/")
async def login(user: User):
    if len(user.password) != 4:
        raise HTTPException(status_code=400, detail="Password must be a 4-digit PIN")
    
    user_in_db = users_db.get(user.username)
    if not user_in_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    if not verify_password(user.password, user_in_db.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid password")
    
    return {"message": "Login successful"}


# Run the FastAPI server with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
