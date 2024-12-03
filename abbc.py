from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext
import torch

# Load Llama 3 Model and Tokenizer
model_name = "meta/llama-3"  # Replace with the correct model name
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Initialize Pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "mps" else -1
)

# Function for General Prompt Response
def generate_response(prompt, max_length=200):
    response = generator(prompt, max_length=max_length, num_return_sequences=1)
    return response[0]["generated_text"]

# Embedding-Based Query Handling with llama-index
def setup_knowledge_base(data_dir):
    """
    Load documents from a directory and create a knowledge base index.
    :param data_dir: Path to the directory containing text documents.
    :return: GPTVectorStoreIndex instance.
    """
    print("Loading documents and creating knowledge base...")
    documents = SimpleDirectoryReader(data_dir).load_data()
    llm_predictor = LLMPredictor(llm=generator)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context
    )
    return index

def query_knowledge_base(index, query):
    """
    Query the knowledge base index.
    :param index: GPTVectorStoreIndex instance.
    :param query: User query string.
    :return: Response from the knowledge base.
    """
    print("Querying knowledge base...")
    response = index.query(query)
    return response.response

# Chatbot with Conversation Memory
class ChatBot:
    def __init__(self):
        self.history = []

    def chat(self, prompt):
        """
        Chat with memory of previous interactions.
        :param prompt: User's input.
        :return: AI's response.
        """
        self.history.append({"role": "user", "content": prompt})
        conversation_context = "\n".join(
            [f"{entry['role']}: {entry['content']}" for entry in self.history]
        )
        response = generate_response(conversation_context, max_length=300)
        self.history.append({"role": "assistant", "content": response})
        return response

# Example Usage
if __name__ == "__main__":
    print("Setting up the AI agent...")

    # General Llama 3 response
    user_prompt = "Explain the concept of neural networks in simple terms."
    response = generate_response(user_prompt)
    print("\nGeneral Response:\n", response)

    # Setup Knowledge Base
    data_directory = "./knowledge_base"  # Replace with the path to your documents
    knowledge_index = setup_knowledge_base(data_directory)

    # Query Knowledge Base
    knowledge_query = "What are the key advantages of AI in healthcare?"
    kb_response = query_knowledge_base(knowledge_index, knowledge_query)
    print("\nKnowledge Base Response:\n", kb_response)

    # Chatbot Interaction
    chatbot = ChatBot()
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat. Goodbye!")
            break
        bot_response = chatbot.chat(user_input)
        print("\nAI:", bot_response)
