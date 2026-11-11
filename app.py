import os
from flask import Flask, render_template, request
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

API_KEY = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

CATEGORIES = "програмування, штучний інтелект, бази даних, комп'ютерні мережі"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()

try:
    chroma_client.delete_collection("lab6_docs")
except Exception:
    pass

collection = chroma_client.create_collection(name="lab6_docs")

DOC_FILES = [
    "doc1_programming.txt",
    "doc2_ai.txt",
    "doc3_db.txt",
    "doc4_networks.txt"
]

docs = []
doc_ids = []

for idx, file_path in enumerate(DOC_FILES, 1):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            docs.append(f.read().strip())
            doc_ids.append(f"doc{idx}")

if docs:
    embeddings = embedding_model.encode(docs).tolist()
    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=doc_ids
    )

TEST_SAMPLES = [
    {
        "id": 1,
        "text": "Python підтримує списки, кортежі та словники для збереження даних.",
        "expected": "програмування"
    },
    {
        "id": 2,
        "text": "Глибоке навчання та нейронні мережі використовуються для аналізу зображень.",
        "expected": "штучний інтелект"
    },
    {
        "id": 3,
        "text": "Команда SELECT виконує вибірку записів із таблиці за заданою умовою.",
        "expected": "бази даних"
    },
    {
        "id": 4,
        "text": "Протокол IP відповідає за адресацію та маршрутизацію пакетів у мережі.",
        "expected": "комп'ютерні мережі"
    },
    {
        "id": 5,
        "text": "Інкапсуляція дозволяє приховати внутрішню реалізацію об'єкта у класі.",
        "expected": "програмування"
    },
    {
        "id": 6,
        "text": "Алгоритми машинного навчання виявляють приховані патерни у великих масивах даних.",
        "expected": "штучний інтелект"
    },
    {
        "id": 7,
        "text": "Первинний ключ використовується для унікальної ідентифікації кожного рядка в таблиці.",
        "expected": "бази даних"
    }
]


def run_prompt_engineering(text):
    prompt = f"""
Відповідай суворо українською мовою!
Визнач категорію наведеного тексту.
Можливі категорії: {CATEGORIES}.
Поверни ЛИШЕ назву категорії (одне-два слова), без розділових знаків та пояснень.

Текст: {text}
"""
    res = client.models.generate_content(model="gemini-3.5-flash-lite", contents=prompt)
    return res.text.strip().lower()

def run_rag(text):
    q_emb = embedding_model.encode([text]).tolist()[0]
    retrieved = collection.query(query_embeddings=[q_emb], n_results=1)['documents'][0][0]
    
    prompt = f"""
Відповідай суворо українською мовою!
Використовуючи наведений контекст, визнач категорію тексту.
Можливі категорії: {CATEGORIES}.
Поверни ЛИШЕ назву категорії.

Контекст: {retrieved}
Текст: {text}
"""
    res = client.models.generate_content(model="gemini-3.5-flash-lite", contents=prompt)
    return res.text.strip().lower(), retrieved

def run_fine_tuning_simulation(text):
    prompt = f"""
Відповідай суворо українською мовою!
Класифікуй текст за однією з категорій: {CATEGORIES}.

Приклади:
Текст: Змінні зберігають значення у пам'яті. -> Категорія: програмування
Текст: Алгоритми машинного навчання виявляють патерни у даних. -> Категорія: штучний інтелект
Текст: Реляційна база даних зберігає дані у вигляді таблиць. -> Категорія: бази даних
Текст: Маршрутизатор передає пакети між різними мережами. -> Категорія: комп'ютерні мережі

Текст: {text}
Категорія:"""
    res = client.models.generate_content(model="gemini-3.5-flash-lite", contents=prompt)
    return res.text.strip().lower()


@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    pe_res = ""
    rag_res = ""
    rag_context = ""
    ft_res = ""

    if request.method == "POST":
        query = request.form.get("query")
        
        pe_res = run_prompt_engineering(query)
        rag_res, rag_context = run_rag(query)
        ft_res = run_fine_tuning_simulation(query)

    return render_template(
        "index.html",
        query=query,
        pe_res=pe_res,
        rag_res=rag_res,
        rag_context=rag_context,
        ft_res=ft_res,
        samples=TEST_SAMPLES
    )

if __name__ == "__main__":
    app.run(debug=True, port=5001)