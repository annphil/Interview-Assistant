from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(question, response):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize and encode the question and response
    question_tokens = tokenizer.encode(question, add_special_tokens=True)
    response_tokens = tokenizer.encode(response, add_special_tokens=True)

    # Convert token IDs to tensors
    question_tensor = torch.tensor([question_tokens])
    response_tensor = torch.tensor([response_tokens])

    # Forward pass through the BERT model
    with torch.no_grad():
        question_embedding = model(question_tensor)[0][:, 0, :].numpy()  # Take the [CLS] token embedding
        response_embedding = model(response_tensor)[0][:, 0, :].numpy()
    
    print("question_tokens: ",question_tokens, "\nquestion_tensor: " , question_tensor, "\nquestion_embedding: ", question_embedding)

    # Calculate cosine similarity
    similarity_score = cosine_similarity(question_embedding, response_embedding)[0][0]

    return similarity_score

from openai import AzureOpenAI

ENDPOINT = "https://polite-ground-030dc3103.4.azurestaticapps.net/api/v1"

API_VERSION = "2024-02-01"
MODEL_NAME = "gpt-35-turbo"

client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION,
)

# Define interview questions
question = {"role": "user", "content": "What is your greatest strength?"}

# Generate responses for interview questions
completion = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[{"role": "system", "content": "You are an interviewee who is a fresher who has to answer the following questions."}, question],
)
assistant_response = completion.choices[0].message.content

print(assistant_response)

# Example usage
# question = "Are you a self-motivator?"
# response = "Absolutely. For me, internal motivation works far more than external motivation ever could."
# question = "What is LSTM?"
response = "Hardwork and smartwork. Not hardwork, competitive mentality"

similarity_score = calculate_similarity(assistant_response, response)
print("Similarity Score:", similarity_score)
