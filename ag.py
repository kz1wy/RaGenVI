import json
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# Initialize embeddings for vietnamese text processing
model_name = 'keepitreal/vietnamese-sbert'
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Load data
with open('data/processed/context_question.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Semantic chunking
text_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type='percentile', breakpoint_threshold_amount=90)

def json_to_documents(json_data):
    documents = []
    for article in json_data:
        article_title = article["title"]
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            docs = text_splitter.create_documents(context)
            for doc in docs:
                doc.metadata = {"title": article_title}
                doc.page_content = context
                documents.append(doc)
    return documents

documents = json_to_documents(data["data"])

# Initialize Chroma database
db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Initialize LLM
llm = ChatOllama(model="llama3.2", temperature=0, max_tokens=1024)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# Query rewriting
query_rewrite_template = """
You are an AI assistant tasked with improving user queries to be more specific and clear, especially when the original query is vague or unclear.
When rewriting, or creating infer the user's intent, remove ambiguity, and make the query actionable based on the given context.
Do not overcomplicate or add irrelevant details. Keep the rewritten query clear and focused. Make sure the question is answeralbe.

Only respond with the rewritten query, in the same language as the original query.

Original query: {original_query}
Context: {context}

Rewritten query:
"""

query_rewrite_prompt = PromptTemplate(
    input_variables=["original_query", "context"],
    template=query_rewrite_template
)
re_write_chain = query_rewrite_prompt | llm

def rewrite_query(original_query):
    retrieved_context = retriever.get_relevant_documents(original_query)
    context = retrieved_context[0].page_content
    response = re_write_chain.invoke({"original_query": original_query, "context": context})
    return response.content

# Step-back prompting
step_back_template = """
You are an AI assistant tasked with generating broader, more general queries to increase diversity and explore wider possibilities.
Based on the original query, create a step-back query that is more open-ended and general, while considering the given context. Ensure the step-back query allows for diverse responses or interpretations and answeralbe.

Only respond with the step-back query, in the same language as the original query.

Original query: {original_query}
Context: {context}

Step-back query:
"""

step_back_prompt = PromptTemplate(
    input_variables=["original_query", "context"],
    template=step_back_template
)

step_back_chain = step_back_prompt | llm

def generate_step_back_query(original_query):
    retrieved_context = retriever.get_relevant_documents(original_query)
    context = retrieved_context[0].page_content
    response = step_back_chain.invoke({"original_query": original_query, "context": context})
    return response.content

# QA Chain
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an assistant for a question-answering task. "
        "Your task is to extract relevant information from the provided context and provide a concise, clear answer. "
        "Ensure your answer is brief and directly addresses the core of the question, short and enough information"
        "Just give the answer, do not disturb with no relevance information"
        "The answer should be in the same language as the input.\n"
        "Structure the response as follows:\n"
        "answer: [Your concise answer]\n"
        "Context: {context}"
    ),
    (
        "human",
        "Question: {question}"
    )
])

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

def process_rag_output(output):
    question = output["query"]
    answer = output["result"].replace("answer: ", "")
    return (question, answer)

def answer_generation(data):
    new_data = []
    for document in tqdm(data, desc="Generating answer"):
        new_document = {
            "id": document["id"],
            "context": document["context"],
            "questions": {
                "beam_search": [],
                "sampling": []
            }
        }

        # Process beam_search questions
        for question in document["questions"]["beam_search"]:
            rewritten_query = rewrite_query(question)
            output = process_rag_output(qa_chain.invoke(rewritten_query))
            new_document["questions"]["beam_search"].append({
                "question": output[0],
                "answer": output[1]
            })

        # Process sampling questions
        for question in document["questions"]["sampling"]:
            step_back_query = generate_step_back_query(question)
            output = process_rag_output(qa_chain.invoke(step_back_query))
            new_document["questions"]["sampling"].append({
                "question": output[0],
                "answer": output[1]
            })

        new_data.append(new_document)

    return new_data

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

# Main execution
if __name__ == "__main__":
    processed_data = answer_generation(data)
    save_to_json(processed_data, 'data/processed/context_question_answer.json')
