import re
import nltk
import torch
from nltk.corpus import stopwords
from transformers import pipeline

# Ensure required NLTK resources are loaded quietly
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

STOP_WORDS = set(stopwords.words("english"))

# Initialize DialoGPT text generation pipeline
chatbot_pipeline = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-medium",
    torch_dtype=torch.float32
)


def preprocess_rule_input(text: str) -> str:
    """Basic cleaning for rule matching (lowercasing and punctuation removal)."""
    text = text.lower()
    return re.sub(r"[^a-z0-9\s]", "", text).strip()


def simple_chatbot(user_input: str) -> str | None:
    """Rule-based lookup for simple conversational intents."""
    cleaned = preprocess_rule_input(user_input)

    responses = {
        "hello": "Hi there! How can I assist you?",
        "how are you": "I'm doing great! How about you?",
        "bye": "Goodbye! Have a wonderful day!",
        "thanks": "You're welcome! Let me know if you need anything else."
    }

    for key, response in responses.items():
        if key in cleaned:
            return response
    return None


def ai_chatbot(user_input: str) -> str:
    """Generates dynamic AI response using DialoGPT and strips the input prompt."""
    output = chatbot_pipeline(
        user_input,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=50256
    )
    
    generated_text = output[0]["generated_text"]
    
    # Strip out the initial user prompt from the generated response
    if generated_text.startswith(user_input):
        generated_text = generated_text[len(user_input):].strip()
        
    return generated_text if generated_text else "I'm not sure how to respond to that."


def chatbot_system():
    """Main execution loop for hybrid chatbot."""
    print("Chatbot: Hello! Type 'exit' to stop.")
    
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
            
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye! 👋")
            break

        # 1. Try rule-based engine first
        rule_response = simple_chatbot(user_input)
        
        if rule_response:
            print(f"Chatbot: {rule_response}")
        else:
            # 2. Fall back to AI-powered generation
            ai_response = ai_chatbot(user_input)
            print(f"Chatbot: {ai_response}")


if __name__ == "__main__":
    chatbot_system()
