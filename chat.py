from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import json
from pprint import pprint

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# pdf_paths = ["./public/cd430f5d18c356cacfbec5c0b48c91e7.pdf", "./public/a6f3f34a88ae51c8459f3527eb4888ec.pdf"]
# docs = []

# for path in pdf_paths:
#     loader = PyPDFLoader(path)
#     doc = loader.load()
#     docs.extend(doc)

# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size=1000,
#     chunk_overlap=100,
   
# )

# split_docs = text_splitter.split_documents(documents=docs)

embedder = GoogleGenerativeAIEmbeddings(
    google_api_key=api_key,
    model="models/text-embedding-004"
    )

# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="12thChemistry",
#     embedding=embedder
# )

# vector_store.add_documents(documents=split_docs)
# print("done")


retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="12thChemistry",
    embedding=embedder
)
# query = "Which unit of concentration  is useful in relating the concentration of a solution with its vapor pressure?"
# search_result = retriever.similarity_search(
#     query=query
# )

# context = "\n\n".join([doc.page_content for doc in search_result])



# messages = [
#     { "role": "system", "content": system_prompt }
# ]



# response = client.chat.completions.create(
#     model='gemini-2.0-flash',
#     response_format={"type": "json_object"},
#     messages=messages
# )
# parsed_response = json.loads(response.choices[0].message.content)
# print(parsed_response["content"])


chat_history = []

while True:
    user_query = input('> ')



    #Parallel Fan Out Query
    system_prompt_generating_multiple_prompts = """
    You are a helpful AI assistant that generates exactly 3 to 5 alternative prompts based on a user's original query.

    Your goal is to:
    - Dig deeper into specific aspects of the original question, or
    - Broaden the scope to include related concepts.

    📌 Rules:
    1. Always return 3 to 5 prompts.
    2. Do not repeat or slightly rephrase the original query.
    3. Each prompt should offer a unique perspective — either more specific or more general — to support diverse information retrieval.
    4. Focus on factual, educational, and logical variations.
    5. Return output strictly in JSON array format as shown in the schema

    📤 Output Format:
    [
        {
            "prompt": "string"
        },
        {
            "prompt": "string"
        },
        {
            "prompt": "string"
        }
    ]

    🎓 Example:
    Original Query: "What are chemical reactions?"

    Output:
    [
        {"prompt": "What are the types of chemical reactions studied in Class 12 Chemistry?"},
        {"prompt": "How do chemical reactions involve changes in energy?"},
        {"prompt": "What is the role of catalysts in chemical reactions?"}
    ]
    """
    messages1 = [
    { "role": "system", "content": system_prompt_generating_multiple_prompts }
    ]

    messages1.append({ "role": "user", "content": user_query })

    response1 = client.chat.completions.create(
        model='gemini-2.0-flash',
        response_format={"type": "json_object"},
        messages=messages1
    )

    llm_generated_prompts = json.loads(response1.choices[0].message.content)
    # print("LLM generated similar prompts:")
    # pprint(llm_generated_prompts)








    # Step Back Prompting
    system_prompt_generating_step_back_prompt = """
    You are an intelligent assistant specialized in generating 'step back' prompts. generate 4 step back prompts for the given user query

    🎯 Objective:
    Generate a higher-level prompt that zooms out to the broader concept or topic from which the user's specific question originates. This helps in understanding the foundational context and conceptual framework.

    📜 Rules:
    1. Your output must strictly follow the JSON schema.
    2. Analyze the original question and identify the larger domain or subject area it belongs to.
    3. Abstract the underlying principles or categories that the specific question fits into.
    4. Phrase the step-back prompt as a general, educational question that can help set the stage for understanding the original query.

    📤 Output Format:
    {
        "prompt": "string"
    }

    🎓 Examples (based on Class 12 Chemistry):
    User prompt: "What is the difference between SN1 and SN2 reactions?"
    Step back prompt: "What are nucleophilic substitution reactions and how are they classified?"

    User prompt: "How does the boiling point of alcohols change with chain length?"
    Step back prompt: "What factors affect the boiling points of organic compounds?"

    User prompt: "Why does glucose not give Schiff's test?"
    Step back prompt: "What are the chemical tests used to identify functional groups in carbohydrates?"
    """

    messages2 = [
    { "role": "system", "content": system_prompt_generating_step_back_prompt }
    ]

    messages2.append({ "role": "user", "content": user_query })

    response2 = client.chat.completions.create(
        model='gemini-2.0-flash',
        response_format={"type": "json_object"},
        messages=messages2
    )

    step_back_prompt = json.loads(response1.choices[0].message.content)
    # print("Step back prompt:")
    # pprint(step_back_prompt)

    all_prompts = llm_generated_prompts + step_back_prompt
    # print("All prompts:")
    # pprint(all_prompts)









    # Similarity Search to find relevant chunks
    chunks_list = list()
    total_chunks_counter = 0
    for prompt in all_prompts:
        relevant_chunks = retriever.similarity_search(
            query=prompt['prompt']
        )
        chunks_list.append(relevant_chunks)

    





    # Reciprocal Rank Fusion of chunks
    doc_scores = {}
    k = 60 # k: A constant used for smoothing the reciprocal rank values

    for ranking in chunks_list:
        for idx, doc in enumerate(ranking):
            doc_id = doc.metadata["_id"]
            rr = 1 / (idx + 1 + k)
            if doc_id in doc_scores:
                doc_scores[doc_id] += rr
            else:
                doc_scores[doc_id] = rr

    fused_ranking = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)




    sorted_relevant_chunk_list = list()
    for chunk_id, rank_score in doc_scores.items():

        for chunks in chunks_list:
            for chunk in chunks:
                if chunk_id in chunk.metadata["_id"]:
                    sorted_relevant_chunk_list.append(chunk)    


    top_relevant_chunks = sorted_relevant_chunk_list[:5]
    context = ""

    for doc in top_relevant_chunks:
        context += doc.page_content + "\n\n"

    






    
    # Finally Generating response based on the most relevant context
    system_prompt = f"""
    You are an intelligent AI assistant designed to answer user queries using the given context. Your job is to extract relevant information *strictly* from the context and respond in a specific JSON format.

    The user will ask a question, and you must answer it only if the context includes the necessary information. If the answer is partially available, explain that. If not at all, state clearly.

    Instructions:
    - Focus on concrete facts from the context (e.g., numbers, dates, durations, names).
    - Do not hallucinate or make assumptions.
    - If the context provides a value like "8 months 6 hrs/week", interpret and return it.
    - If no answer is found, return a polite message.
    - Go in detail explaining the answer and the context.
    - Format your answer in a readable and well-structured way with paragraphs and line breaks for clarity.
    - First write whatever you know about the user query from the given context and then write that you dont know about something if it is not in the context and mention that it isn't there in the context.

    Output Format (strictly):
    {{
    "step": "answer",
    "content": "your complete answer based on the context"
    }}

    Example 1:
    Query: "What are biomolecules?"
    Context: "...Course Instructors: Sunny Savita, Krish Naik..."
    Output:
    {{
    "step": "answer",
    "content": "Biomolecules are chemical compounds found in living organisms that play crucial roles in sustaining life. They are broadly categorized into two types based on their molecular weight:​

    Micromolecules (Low Molecular Weight Biomolecules): These have molecular weights less than 1000 daltons and include simple compounds like amino acids, sugars, and nucleotides.​

    Macromolecules (High Molecular Weight Biomolecules): These have molecular weights above 10,000 daltons and are typically polymeric, meaning they are composed of repeating monomer units. Examples include proteins, nucleic acids (DNA and RNA), polysaccharides, and lipids.
    }}

    Context to refer:
    {context}
    """
    messages = [
        { "role": "system", "content": system_prompt },
        *chat_history,
        { "role": "user", "content": user_query }
    ]

    response = client.chat.completions.create(
    model='gemini-2.0-flash',
    response_format={"type": "json_object"},
    messages=messages
    )
    parsed_response = json.loads(response.choices[0].message.content)
    messages.append({ "role": "assistant", "content": parsed_response["content"] }) 
    print(parsed_response["content"])
    chat_history.append({ "role": "user", "content": user_query })
    chat_history.append({ "role": "assistant", "content": parsed_response["content"] })
    








