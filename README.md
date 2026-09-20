# Hybrid AI Conversational Agent

A Python-based hybrid chatbot that combines the speed and reliability of deterministic rule-based NLP with the dynamic conversational capabilities of deep learning.

By routing queries through a tiered pipeline, the system handles common intents instantly using regular expressions and NLTK text processing, while offloading complex, open-ended interactions to Microsoft's DialoGPT via the Hugging Face `transformers` library.

## Features
* **Tiered Query Routing:** Prioritizes exact-match rule generation for standard intents, reducing unnecessary LLM inference costs and latency.
* **Generative Fallback:** Leverages `microsoft/DialoGPT-medium` for contextual, open-ended text generation when user inputs fall outside predefined rules.
* **Robust Text Preprocessing:** Utilizes NLTK tokenization and regular expressions to strip noise and normalize user inputs without losing grammatical intent.
* **Local CPU/GPU Inference:** Runs entirely locally using PyTorch, with adjustable sampling parameters (Top-K, Top-p) for diverse response generation.

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mostafa-8811/hybrid-ai-chatbot.git](https://github.com/mostafa-8811/hybrid-ai-chatbot.git)
   cd hybrid-ai-chatbot
