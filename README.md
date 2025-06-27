# Implementation-of-Machine-Translation

## Aim:
To implement English-to-French translation using a pre-trained Transformer model (Helsinki-NLP/opus-mt-en-fr) from Hugging Face's transformers library.

## Requirements:
**Software Requirements:**
Python 3.x

Hugging Face Transformers library

PyTorch
**Hardware Requirements:**
A machine with CPU (or GPU for faster translation)

## Procedure:
Import Libraries:

Import AutoTokenizer and AutoModelForSeq2SeqLM from transformers.

Import torch for tensor manipulation and no_grad operations.

Load Pre-trained Model:

Load the tokenizer and model from Hugging Face (opus-mt-en-fr), a model specialized for English-to-French translation.

Tokenization and Translation:

Tokenize input English text.

Generate French translation using the model's generate method.

Decode the output tokens back into a human-readable string.

Test the Model:

Define multiple English sentences.

Pass each sentence to the translation function and print the translated output.

## Program:
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load pre-trained model and tokenizer for English-to-French translation
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate_text(text: str, max_length: int = 40) -> str:
    # Tokenize input and create input tensor
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)

    # Generate translation
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # Decode translation
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# English input sentences
english_sentences = [
    "Hello, how are you?",
    "This is an experiment in machine translation.",
    "Transformers are powerful models for natural language processing tasks.",
    "Can you help me with my homework?",
    "I love learning new languages."
]

# Translate and print results
for sentence in english_sentences:
    translation = translate_text(sentence)
    print(f"Original: {sentence}")
    print(f"Translated: {translation}\n")
  ```
## Output:
```vbnet
Original: Hello, how are you?
Translated: Bonjour, comment ça va ?

Original: This is an experiment in machine translation.
Translated: Ceci est une expérience de traduction automatique.

Original: Transformers are powerful models for natural language processing tasks.
Translated: Les transformateurs sont des modèles puissants pour les tâches de traitement du langage naturel.

Original: Can you help me with my homework?
Translated: Peux-tu m'aider avec mes devoirs ?

Original: I love learning new languages.
Translated: J'aime apprendre de nouvelles langues.
```
## Result:
The pre-trained Transformer model accurately translated English sentences into grammatically correct French, demonstrating the effectiveness of Transformer-based architectures in machine translation
