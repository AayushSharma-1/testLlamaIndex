from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini
from pinecone import Pinecone
from llama_index.core.memory import ChatMemoryBuffer
import os
def initialize_components(api_key, model_name):
    """
    Initialize Pinecone, Gemini, and other necessary components.
    """
    llm = Gemini(api_key=api_key, model_name=model_name)
    pc = Pinecone(api_key=api_key)
    return llm, pc

def configure_settings(embedding_model):
    """
    Configure settings for embeddings and LLM.
    """
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

def setup_chat_engine(pc_index, chat_mode, memory, system_prompt, verbose=False):
    """
    Set up the chat engine with required parameters.
    """
    vector_store = PineconeVectorStore(pc_index)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    chat_engine = index.as_chat_engine(
        chat_mode=chat_mode,
        memory=memory,
        system_prompt=system_prompt,
        verbose=verbose
    )
    return chat_engine

def handle_user_input(chat_engine):
    """
    Handle user input and execute the chat.
    """
    while True:
        user_input = input("Enter Query: ")
        response = chat_engine.chat(user_input)
        print(response)

def main():
    GOOGLE_API_KEY = "AIzaSyCMAvB0-ehycivbI10OaaqY9WNXUe20U7U"
    MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    SYSTEM_PROMPT = """You are an expert informator system about Lucknow,
    I'll give you question and context and you'll return the answer in a sweet and sarcastic tone. 
    You will use Hum instead of main. Your name is Lallan. 
    The full form of Lallan is 'Lucknow Artificial Language and Learning Assistance Network'. 
    Call only Janab-e-Alaa instead of phrase My dear Friend. 
    Say Salaam Miya! instead of Greetings. 
    If you do not find the context suitable then do not hallucinate the answer You can Answer the Question on you own our you can Just Say You dont know!!
    if you enconuter question which ask about you like hello, Hii or a normal conversation you can simply talk then there's no need to answer from the context"""
    api_key = os.getenv("PINECONE_API_KEY")
    
    llm, pc = initialize_components(GOOGLE_API_KEY, MODEL_NAME)
    configure_settings(MODEL_NAME)
    
    pc_index = pc.Index("quickstart")
    memory = ChatMemoryBuffer.from_defaults(token_limit=5000)
    
    chat_engine = setup_chat_engine(
        pc_index, "context", memory, SYSTEM_PROMPT, verbose=True
    )
    
    handle_user_input(chat_engine)

if __name__ == "__main__":
    main()
