# -----------------------
# Importações
# -----------------------

import os
import time
import chromadb
import google.generativeai as genai
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Biblioteca RAGAS
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from langchain_google_genai import ChatGoogleGenerativeAI

# -----------------------
# Configurações
# -----------------------

GOOGLE_API_KEY = "AIzaSyA9-k3Zr9cZTJCHrAwA9wQKNJgQB4stfGI" 

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    normalize=True,
)
Settings.llm = None

persist_dir = "db_ihc"

# -----------------------
# Indexação dos documentos
# -----------------------

documents = SimpleDirectoryReader(
    input_dir="./arquivosFormatados",
    file_metadata=lambda x: {
        "file_name": os.path.basename(x),
        "file_path": x
    }
).load_data()

text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
all_nodes = text_splitter.get_nodes_from_documents(documents)

chroma_client = chromadb.PersistentClient(path=persist_dir)
chroma_collection = chroma_client.get_or_create_collection(name="docs_ihc")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(all_nodes, storage_context=storage_context, show_progress=True)
storage_context.persist(persist_dir=persist_dir)
query_engine = index.as_query_engine(similarity_top_k=8)

# -----------------------
# Função para gerar resposta com Gemini
# -----------------------

def generate_response_with_gemini(prompt, max_retries=3):
    genai.configure(api_key=GOOGLE_API_KEY)
    
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-lite",
        generation_config=generation_config
    )

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if response and hasattr(response, 'text'):
                return response.text.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"Erro: {str(e)}")
    return "Erro ao gerar resposta."

# -----------------------
# Definição das perguntas e respostas esperadas
# -----------------------

sample_queries = [
    "O que é usabilidade na área de Interação Humano-Computador?",
    "Quais são os principais tipos de protótipos utilizados no design de interfaces?",
    "Como a teoria da carga cognitiva influencia o design de interfaces complexas?",
    "Como o aplicativo Blind Magic promove a acessibilidade para jogadores com deficiência visual no jogo Magic: The Gathering?"
]

expected_responses = [
    "Usabilidade é a facilidade com que os usuários conseguem usar uma interface para realizar suas tarefas. Envolve critérios como eficiência, eficácia, satisfação e aprendizado.",
    "Os principais tipos de protótipos são os de baixa fidelidade (como wireframes), média fidelidade (mockups com navegação) e alta fidelidade (protótipos interativos próximos ao produto final).",
    "A teoria da carga cognitiva destaca a importância de reduzir a carga mental dos usuários durante a interação. Interfaces bem projetadas minimizam informações desnecessárias e organizam elementos visuais para favorecer o processamento eficiente, especialmente em sistemas complexos.",
    "O Blind Magic é um aplicativo acessível desenvolvido para permitir que pessoas com deficiência visual joguem Magic: The Gathering. Ele utiliza códigos QR únicos em cada carta, que, ao serem escaneados por um smartphone, fornecem feedback auditivo e tátil ao usuário."
]

# -----------------------
# Geração das respostas e montagem do dataset
# -----------------------

dataset = []

for query, gt_answer in zip(sample_queries, expected_responses):
    print(f"\n🧠 Avaliando: {query}")

    # Obter chunks relevantes
    response_object = query_engine.query(query)

    # Criar contexto
    context_chunks = "\n\n".join(
        f"📄 Fonte: {node.metadata.get('file_name', 'Desconhecida')}\n"
        f"📊 Score: {node.score:.2f}\n"
        f"🔍 Conteúdo:\n{node.node.text[:1000]}..."
        for node in response_object.source_nodes
    )

    print("\n📚 Contexto usado no prompt:\n")
    print(context_chunks)

    # Criar prompt
    prompt = f"""
Você é um especialista em IHC (Interação Humano-Computador) com vasta experiência acadêmica e prática.

[INSTRUÇÕES]
1. Analise cuidadosamente a pergunta e o contexto fornecido.
2. Se o contexto contiver informações relevantes, baseie sua resposta nessas informações.
3. Se o contexto for insuficiente, use seu conhecimento de IHC.
4. Não mencione diretamente os \"textos fornecidos\".

[CONTEXTO]
{context_chunks}

Pergunta: {query}
"""

    model_response = generate_response_with_gemini(prompt)

    # Montar para o dataset
    retrieved_contexts = [node.node.text for node in response_object.source_nodes]

    dataset.append({
        "user_input": query,
        "retrieved_contexts": retrieved_contexts,
        "response": model_response,
        "reference": gt_answer
    })

# -----------------------
# Avaliação usando RAGAS
# -----------------------

# Usar o Gemini para avaliação também
evaluator_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )
)

# Construir dataset para avaliação
evaluation_dataset = EvaluationDataset.from_list(dataset)

# Avaliar
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm
)

print("\n🏁 Resultados da Avaliação:")
print(result)
