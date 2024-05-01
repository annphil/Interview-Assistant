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

# Example usage
# question = "Are you a self-motivator?"
# response = "Absolutely. For me, internal motivation works far more than external motivation ever could."
question = "What is LSTM?"
response = "How do u do"

similarity_score = calculate_similarity(question, response)
print("Similarity Score:", similarity_score)
