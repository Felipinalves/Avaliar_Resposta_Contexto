import os
import time
import chromadb
import google.generativeai as genai
from ragchecker.integrations.llama_index import response_to_rag_results
from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

# -----------------------
# Configurações do sistema
# -----------------------

# Configuração do embedding
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    normalize=True,
)
Settings.llm = None  # Desliga o LLM interno

# Caminho do banco
persist_dir = "db_ihc"

# Leitura dos documentos
documents = SimpleDirectoryReader(
    input_dir="./arquivosFormatados",
    file_metadata=lambda x: {
        "file_name": os.path.basename(x),
        "file_path": x
    }
).load_data()

# Quebra dos documentos
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
all_nodes = text_splitter.get_nodes_from_documents(documents)

# Armazenamento com ChromaDB
chroma_client = chromadb.PersistentClient(path=persist_dir)
chroma_collection = chroma_client.get_or_create_collection(name="docs_ihc")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Criação do índice
index = VectorStoreIndex(all_nodes, storage_context=storage_context, show_progress=True)
storage_context.persist(persist_dir=persist_dir)
query_engine = index.as_query_engine(similarity_top_k=8)

# -----------------------
# Função para gerar resposta com Gemini
# -----------------------

GOOGLE_API_KEY = ""

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
# Perguntas para avaliação
# -----------------------

test_questions = [
    {
        "query_id":"000",
        "query": "O que é usabilidade na área de Interação Humano-Computador?",
        "gt_answer": "Usabilidade é a facilidade com que os usuários conseguem usar uma interface para realizar suas tarefas. Envolve critérios como eficiência, eficácia, satisfação e aprendizado."
    },
    {
        "query_id":"001",
        "query": "Quais são os principais tipos de protótipos utilizados no design de interfaces?",
        "gt_answer": "Os principais tipos de protótipos são os de baixa fidelidade (como wireframes), média fidelidade (mockups com navegação) e alta fidelidade (protótipos interativos próximos ao produto final)."
    },
    {
        "query_id":"002",
        "query": "Como a teoria da carga cognitiva influencia o design de interfaces complexas?",
        "gt_answer": "A teoria da carga cognitiva destaca a importância de reduzir a carga mental dos usuários durante a interação. Interfaces bem projetadas minimizam informações desnecessárias e organizam elementos visuais para favorecer o processamento eficiente, especialmente em sistemas complexos."
    },
    {
        "query_id":"003",
        "query": "Como o aplicativo Blind Magic promove a acessibilidade para jogadores com deficiência visual no jogo Magic: The Gathering?",
        "gt_answer": "O Blind Magic é um aplicativo acessível desenvolvido para permitir que pessoas com deficiência visual joguem Magic: The Gathering. Ele utiliza códigos QR únicos em cada carta, que, ao serem escaneados por um smartphone com sistema Android 4.4 ou superior, fornecem feedback auditivo e tátil ao usuário. O aplicativo foi desenvolvido através de um processo iterativo de co-design com um usuário cego, seguindo diretrizes de acessibilidade do Android e incorporando o leitor de tela TalkBack para garantir uma experiência inclusiva."
    },
    
]

# -----------------------
# Avaliação com RAGChecker
# -----------------------

results = []

for idx, item in enumerate(test_questions):
    print(f"\n🧠 Avaliando: {item['query']}")

    # Obter chunks relevantes
    response_object = query_engine.query(item["query"])

    # Criar prompt com contexto + pergunta
    context_chunks = "\n\n".join(
        f"📄 Fonte: {node.metadata.get('file_name', 'Desconhecida')}\n"
        f"📊 Score: {node.score:.2f}\n"
        f"🔍 Conteúdo:\n{node.node.text[:1000]}..."
        for node in response_object.source_nodes
    )
    
    # Mostrar o contexto usado
    print("\n📚 Contexto usado no prompt:\n")
    print(context_chunks)

    prompt = f"""Você é um especialista em IHC (Interação Humano-Computador) com vasta experiência acadêmica e prática.

[INSTRUÇÕES]
1. Analise cuidadosamente a pergunta e o contexto fornecido.
2. Se o contexto contiver informações relevantes para a pergunta, baseie sua resposta principalmente nessas informações.
3. Se o contexto for insuficiente ou não abordar diretamente a pergunta, forneça uma resposta baseada em seu conhecimento geral de IHC, sem mencionar a ausência de informações no contexto.
4. Não faça referências diretas aos "textos fornecidos" ou "artigos" na sua resposta.

[FORMATO]
- Use português brasileiro formal
- Mantenha termos técnicos consolidados em inglês quando apropriado
- Estruture sua resposta em parágrafos claros e concisos
- Inclua exemplos práticos quando relevante
- Apresente diferentes perspectivas quando apropriado

[CONTEXTO]
{context_chunks}

Pergunta: {item['query']}
"""

    # Gerar resposta
    model_response = generate_response_with_gemini(prompt)

    # Adaptar para RAGChecker
    rag_result = response_to_rag_results(
        query=item["query"],
        gt_answer=item["gt_answer"],
        response_object=response_object
    )

    rag_result['query_id'] = item['query_id']
    rag_result['response'] = model_response

    results.append(rag_result)

# Avaliar
rag_results = RAGResults.from_dict({"results": results})
evaluator = RAGChecker(
    extractor_name="ollama/llama2",
    checker_name="ollama/llama2",
    batch_size_extractor=32,
    batch_size_checker=32,
)

evaluator.evaluate(rag_results, all_metrics)

# Mostrar resultado
print("\n📊 Resultados detalhados:")
for r in rag_results.results:
    print(f"\n🔎 Pergunta: {r.query}")
    print(f"✅ Resposta correta esperada: {r.gt_answer}")
    print(f"🤖 Resposta do modelo: {r.response}")
    print(f"📈 Métricas: {r.metrics}")
