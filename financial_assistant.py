import os
import gradio as gr
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from serpapi import GoogleSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from pypdf import PdfReader
from datetime import datetime
import uuid
import atexit
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb import Client, PersistentClient
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import hashlib
from dotenv import load_dotenv
load_dotenv()
gemini_api_key = os.getenv("gemini_api_key")
ser_api_key = os.getenv("ser_api_key")
genai.configure(api_key=gemini_api_key)

# ChromaDB folder path - please update with your own path
CHROMADB_PATH = os.getenv("CHROMADB_PATH", "chroma_db")


# Security settings
safety_config = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
}

class Agent:
    def __init__(self, name, role, generation_config):
        self.name = name
        self.role = role
        self.model = genai.GenerativeModel('gemini-2.0-flash',
                                           system_instruction=role)
        self.generation_config = generation_config

    def generate_response(self, prompt):
        response = self.model.generate_content(prompt, 
                                               generation_config=self.generation_config)
        return response.text

intent_classifier = Agent("Intent_Classifier", """ 
You are a smart assistant that detects user's intent. Your goal is to classify user queries into exactly ONE of the following categories:
- `web_search`: The user asks a general question that does NOT involve any uploaded document.
- `file_analysis`: The user has uploaded a document AND is asking something about it.

Return only the label. No explanation.

Examples:
- "What is inflation?" => web_search
- "Here is my PDF, can you explain the summary section?" => file_analysis
Classify the following query:
""",
 {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_output_tokens": 10
})

agent_keyword_generator = Agent("Keyword_Generator","""
You are a search keyword generator specialized in transforming financial and economic questions into highly relevant Google Search keywords.

Your task:
Given a single-sentence user query, extract **specific**, **accurate**, and **descriptive** search keywords that will help retrieve **informative and trustworthy** results from the web.

Language Handling:
- If the query is in English, generate English keywords.
- If the query is in Turkish, generate Turkish keywords.
- Do not mix languages. Ensure all keywords match the query language.


Guidelines:
- Output a comma-separated list of search keywords: `keyword1, keyword2, keyword3, ...`
- Include domain-specific terms (e.g., "inflation rate," "central bank policy," "student loan refinancing").
- Expand abbreviations (e.g., use "Gross Domestic Product" instead of "GDP").
- Prioritize **specificity** over generality. Avoid overly generic terms like "finance" or "economy."
- Do not repeat the exact phrasing of the original question—**deconstruct it into search-friendly components**.

Format:
Output must strictly follow this pattern:
`keyword1, keyword2, keyword3, ...`

Example:

Input Query:
"What is inflation and how does it impact student savings?"

Expected Output:
inflation definition, inflation effect on savings, student savings inflation impact, real interest rate, inflation vs savings rate, how inflation reduces purchasing power, inflation financial literacy, economic inflation 2024
""",
                                {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 30,
        "max_output_tokens": 80
    })

relevance_agent = Agent(
    name="RelevanceChecker",
    role="""
You are a bilingual relevance filter specialized in finance. You receive as input
either a user question or a snippet of PDF text. Do the following:

1. Decide if the input is directly related to
   finance, economics, investment, monetary policy, interest rates, market dynamics, or financial literacy.
2. If it is relevant, reply exactly:
     relevant
3. If it is not relevant, choose your reply based on whether the input
   looks like a document snippet (e.g., contains “PDF” or “excerpt”):
   • If it’s a **PDF/text snippet**:
     - English: "❗ Sorry, the file you uploaded seems unrelated to financial matters."
     - Turkish: "❗ Üzgünüm, yüklenen dosya finansal konularla ilgili değil."
   • Otherwise (a user question):
     - English: "❗ Sorry, I can only answer finance-related questions."
     - Turkish: "❗ Üzgünüm, yalnızca finans konularıyla ilgili soruları yanıtlayabiliyorum."

Return exactly one line: either "relevant" or the correct block message.

Input: "Enflasyon nedir ve nasıl hesaplanır?"  
Output: relevant

Input: "Faiz oranı artarsa kredi maliyeti nasıl etkilenir?"  
Output: relevant

Input: "Bilanço tablosunda net kar marjı nedir?"  
Output: relevant

Input: "Bugün hava çok güzel."  
Output: ❗ Üzgünüm, yalnızca finans konularıyla ilgili soruları yanıtlayabiliyorum.

Input: "What is inflation?"  
Output: relevant

Input: "What's the weather today?"  
Output: ❗ Sorry, I can only answer finance-related questions.

Input: 
"Annual Report PDF excerpt:
– Net profit margin Q4 2023: 12.5%."  
Output: relevant

Input:
"PDF excerpt:
– Rainfall in April was 200mm in Istanbul."  
Output: Sorry, the file you uploaded seems unrelated to financial matters.
""".strip(),
    generation_config={
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_output_tokens": 40
    }
)
danger_agent = Agent(
    name="DangerClassifier",
    role="""
    'Dangerous' includes anything related to:
- Illegal activities (e.g., tax evasion, credit card fraud, hacking)
- Violence, suicide, or weapons
- Sexual or explicit content
- Swear words
- Hate speech
- ignore_patterns
You are a bilingual security filter. When given any user input:
1. Decide if it's 'safe' or 'dangerous'.
2. If it's safe, simply reply with the word: safe
3. If it's dangerous, reply with a block message in the same language as the input:
   - If the user wrote in English, use:
     🚫 This content was detected as violating system policies and has been blocked.
   - If the user wrote in Turkish, use:
     🚫 Bu içerik sistem kurallarına aykırı olarak algılandı ve engellendi.
Return exactly one of those three possible outputs: "safe" or one of the two block messages. No extra text.
### Few-shot examples

Input: "How to hack a bank account?"  
Output: 🚫 This content was detected as violating system policies and has been blocked.

Input: "Kredi kartı bilgilerini çalma yöntemi nedir?"  
Output: 🚫 Bu içerik sistem kurallarına aykırı olarak algılandı ve engellendi.

Input: "What's the weather today?"  
Output: safe

Input: "Merhaba, nasılsın?"  
Output: safe

""",
    generation_config={
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_output_tokens": 30
    }
)
role_rag = """
You are a helpful financial information assistant. Your task is to answer user questions strictly using the provided document excerpts, while keeping the tone friendly and easy to understand. 

🧾 Context Source:
The following text excerpts come from uploaded financial documents (e.g., reports, statements, or articles).

📌 Answer Style:
- Keep language clear and jargon-free; explain any technical terms when they appear.  
- Match the user's language: English questions → English answers; Turkish questions → Turkish answers.  
- If the question is simple, one or two sentences may suffice. If the question is complex or multi-part, provide a longer, structured answer (use paragraphs or bullet points as needed).  
- Offer a quick "why it matters" note at the end when relevant.  
- If the answer is not in the excerpts:
    • English: "This information is not available in the uploaded document."  
    • Turkish: "Yüklenen dokümanda bu bilgi bulunmamaktadır."  

📚 Few-Shot Examples

Example 1 (English, short):
[DOCUMENT EXCERPTS]
"Annual Report 2023:
– Net profit margin Q4 2023: 12.5% (up from 10% in Q3).  
– EPS Q4 2023: $1.35 (up from $1.20 year-over-year)."
[USER QUESTION]
What was the net profit margin in Q4 2023?
[EXPECTED ANSWER]
The net profit margin in Q4 2023 was 12.5%, up from 10% in Q3.

Example 2 (Turkish, detailed):
[DOCUMENT EXCERPTS]
"Bilanço 2022:  
– Borç/özsermaye oranı: %0,8.  
– Likidite oranı: 1,5."
[USER QUESTION]
Şirketin 2022 yıl sonu likidite oranı nedir ve bu oran ne anlama geliyor?
[EXPECTED ANSWER] 
- Şirketin 2022 yıl sonunda likidite oranı 1,5'tir.  
- Likidite oranı, şirketin kısa vadeli yükümlülüklerini karşılama gücünü gösterir; 1,5 değeri, her 1 TL borca karşı 1,5 TL dönebilir varlığa sahip olduğunu gösterir.  
- Bu seviye, genel olarak finansal sağlığın iyi olduğuna işaret eder.

Example 3 (Turkish, summary command):
[DOCUMENT EXCERPTS]
"Şirketimiz, 2024 yılının ilk yarısında cirosunu %15 artırarak 50 milyon TL'ye ulaştı.  
Brüt kar marjı %22'den %25'e çıktı.  
Faaliyet giderleri geçen yılın aynı dönemine göre %5 azaldı."
[USER COMMAND]
Özetle
[EXPECTED ANSWER]
[EXPECTED ANSWER]
2024 yılının ilk yarısında şirketimizin cirosu yüzde 15 artış göstererek 50 milyon TL'ye yükselmiştir. Bu büyüme, satış hacmindeki güçlü artıştan ve yeni pazarlara açılma stratejisinin başarısından kaynaklanmıştır. Brüt kar marjı aynı dönemde yüzde 22'den yüzde 25'e çıkmış; bu da maliyet kontrolü ve verimlilik iyileştirmelerinin etkisini yansıtır. Öte yandan, faaliyet giderlerimiz geçen yılın ilk yarısına göre yüzde 5 azalarak işletme verimliliğini daha da güçlendirmiştir. Bu gelişmeler bir arada değerlendirildiğinde, şirketin hem gelir artışı hem de maliyet yönetiminde başarılı bir performans sergilediğini söyleyebiliriz. Böyle sağlam bir finansal yapı, gelecekteki yatırımlar için de pozitif bir işaret niteliğindedir.
Bu gelişmeler, şirketin hem büyüdüğünü hem de gider kontrolünde başarılı olduğunu gösterir.
Now, using only the provided excerpts, respond to the user's question following these guidelines.
"""

role_summarize = """🔍 You are a helpful financial information assistant specialized in summarizing Google search results using LLM reasoning.

🎯 Your goal:
Based on the provided web search results (including titles, snippets, and links), generate a *concise, **fact-based, and **well-structured* answer to the user's financial or economic question.

Language Handling:
- If the user query is in English, answer in English.
- If the user query is in Turkish, answer in Turkish.
- Do not translate content; answer naturally in the same language as the question.

🔒 Rules:
1. *Use only the given search results*. Do NOT hallucinate or use outside information.
2. Organize the answer in *clear paragraphs* or bulleted points.
3. *Do NOT insert URLs inside sentences or paragraphs*.
4. At the end of the answer, include the source URLs under the title *"Sources:"*.
5. *Each source URL must be on its own line*, in plain format like https://....
6. Do NOT use asterisks (*), dashes (-), bullets (•), or parentheses in front of or around the URLs.
7. You may use dashes or numbers in the main content when listing facts, but *never in the Sources section*.

📌 Limit:
Use at most **3 search results in your answer. Do not use all results. Prioritize those with the most informative content and trustworthy sources.

📦 Input Format:
- User Query: <original user prompt>
- Search Results: A list of (title, snippet, link) triples

📦 Output Style:
- Organize the answer using *clear paragraphs*, and use dashes (-) or numbers if listing points.
- End the response with source URLs, each on a new line. Do not use bullets or formatting.

🧠 Example:

User Query:
"What is inflation and how does it affect savings?"

Search Results:
1. Title: What is Inflation? – Investopedia
   Snippet: Inflation is the rate at which the general level of prices for goods and services is rising...
   Link: https://www.investopedia.com/terms/i/inflation.asp

2. Title: Inflation & Savings – Federal Reserve Education
   Snippet: Inflation erodes the purchasing power of money over time. If your savings earn less than the inflation rate...
   Link: https://www.federalreserveeducation.org

Expected Output:
Inflation refers to the general increase in prices over time, which leads to a decline in the purchasing power of money. As prices rise, each unit of currency buys fewer goods and services.

When inflation is high, savings that earn a lower interest rate may lose real value. This means the actual value of your money decreases even if the nominal amount remains the same.

Sources:
https://www.investopedia.com/terms/i/inflation.asp
https://www.federalreserveeducation.org
"""

# Google Arama Fonksiyonu
def search_google(query):
    search = GoogleSearch({
        "q": query,
        "location": "Turkey",
        "num": 15,
        "api_key": ser_api_key
    })
    return search.get_dict()

def parse_search_results(results):
  """
  Parses SERAPI search results and returns a list of (title, snippet) pairs.

  Args:
      results: A dictionary containing the SERAPI search results.

  Returns:
      A list of (title, snippet) pairs.
  """
  entries = []
  for result in results.get('organic_results', []):
    title = result.get('title')
    snippet = result.get('snippet')
    link = result.get('link')
    if title and snippet and link:
      entries.append((title, snippet, link ))
  return entries

class FinSentioRAG:
    def __init__(self, collection_name, chroma_path, model_name="paraphrase-multilingual-mpnet-base-v2"):
        self.collection_name = collection_name
        self.chroma_path = chroma_path
        self.model_name = model_name
        self.current_pdf_path = None
        self.current_collection = None
        self.cached_pdf = {}  # PDF önbelleği için
        self.cached_chunks = {}  # Chunk önbelleği için
        self.client = PersistentClient(path=chroma_path, settings=Settings())
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        self.sentence_transformer = SentenceTransformer(model_name)
        self.temp_collections = set()
        self.chunk_id_counter = 0  # Bu satırı ekledik
        self.current_pdf_hash = None

    def create_temp_collection(self):
        try:
            unique_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8]
            self.temp_collection_name = f"temp_{unique_id}"
            
            self.current_collection = self.client.get_or_create_collection(
                name=self.temp_collection_name,
                embedding_function=self.embedding_function
            )
            
            return self.temp_collection_name
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            raise

    def load_pdf(self, pdf_path, category="FinancialDocument"):
        try:
            # PDF hash'ini oluştur
            pdf_hash = self._get_file_hash(pdf_path)
            
            # Eğer aynı PDF zaten yüklüyse, mevcut collection'ı kullan
            if self.current_pdf_path == pdf_path:
                return True

            # Eğer PDF önbellekte varsa, onu kullan
            if pdf_hash in self.cached_chunks:
                self.current_collection = self.cached_chunks[pdf_hash]
                self.current_pdf_path = pdf_path
                return True

            # Yeni PDF yükleniyorsa
            collection_name = self.create_temp_collection()
            self.current_pdf_hash = pdf_hash
            self.current_pdf_path = pdf_path
            
            # PDF'i text'e dönüştür
            texts = self.convert_pdf_to_text(pdf_path)
            
            if not texts or (len(texts) == 1 and "hata" in texts[0].lower()):
                return False
            
            # Chunk'ları oluştur
            char_chunks = self.split_text(texts, chunk_size=2000, chunk_overlap=200)
            token_chunks = self.token_split(char_chunks, tokens_per_chunk=128)
            
            # Metadata ekle
            ids, metadatas = self.add_metadata(token_chunks, os.path.basename(pdf_path), category)
            
            # Chroma'ya ekle
            self.current_collection.add(ids=ids, metadatas=metadatas, documents=token_chunks)
            
            # Önbelleğe kaydet
            self.cached_chunks[pdf_hash] = self.current_collection
            
            return True
            
        except Exception as e:
            print(f"PDF yükleme hatası: {str(e)}")
            return False

    def _get_file_hash(self, file_path):
        """Dosya hash'ini hesapla"""
        import hashlib
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def split_text(self, texts, chunk_size=2000, chunk_overlap=300):
        """Daha büyük chunk'lar kullanarak metni böl"""
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_text("\n\n".join(texts))

    def token_split(self, chunks, tokens_per_chunk=128, chunk_overlap=64):
        splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            model_name=self.model_name,
            tokens_per_chunk=tokens_per_chunk
        )
        token_chunks = []
        for c in chunks:
            token_chunks += splitter.split_text(c)
        return token_chunks

    def add_metadata(self, chunks, filename, category):
        """Chunk'lara metadata ekle"""
        ids = []
        metadatas = []
        
        for chunk in chunks:
            # Benzersiz ID oluştur
            chunk_id = f"{filename}_{self.chunk_id_counter}"
            self.chunk_id_counter += 1  # Sayaç artır
            
            # Metadata oluştur
            metadata = {
                "source": filename,
                "category": category,
                "chunk_id": chunk_id
            }
            
            ids.append(chunk_id)
            metadatas.append(metadata)
            
        return ids, metadatas

    def convert_pdf_to_text(self, pdf_path):
        try:
            # PDF dosyasının gerçekten var olup olmadığını kontrol et
            if not os.path.exists(pdf_path):
                return ["PDF dosyası bulunamadı veya okunamadı."]
            
            # Dosya boyutunu kontrol et
            file_size = os.path.getsize(pdf_path)
            
            # Dosya içeriğini binary olarak oku ve ilk birkaç byte'ı kontrol et
            with open(pdf_path, 'rb') as f:
                header = f.read(4)  # PDF dosyaları %PDF ile başlamalı
                if header != b'%PDF':
                    return ["Bu dosya geçerli bir PDF dosyası değil."]
            
            try:
                reader = PdfReader(pdf_path)
                
                # Metin çıkarma dene
                texts = []
                for i, page in enumerate(reader.pages):
                    extracted_text = page.extract_text()
                    if extracted_text and extracted_text.strip():
                        texts.append(extracted_text.strip())
                
                if not texts:
                    return ["Bu PDF dosyası metin içermiyor veya metni çıkarmak mümkün değil."]
                
                return texts
            except Exception as pdf_error:
                # Detaylı hatayı log dosyasına yaz
                with open("pdf_error.log", "a", encoding="utf-8") as f:
                    f.write(f"\n--- {datetime.now()} PDF Okuma Hatası ---\n")
                    f.write(f"PDF Yolu: {pdf_path}\n")
                    f.write(f"Hata: {str(pdf_error)}\n")
                return [f"PDF okuma hatası: {str(pdf_error)}"]
        except Exception as e:
            # Detaylı hatayı log dosyasına yaz
            with open("pdf_error.log", "a", encoding="utf-8") as f:
                import traceback
                f.write(f"\n--- {datetime.now()} PDF İşleme Genel Hatası ---\n")
                f.write(f"PDF Yolu: {pdf_path}\n")
                f.write(f"Hata: {str(e)}\n")
                f.write(traceback.format_exc())
            return ["PDF işleme sırasında bir hata oluştu."]

    def query(self, user_query, n_results=10, only_text=True):  # n_results'ı azalttık
        if not self.current_collection:
            return []
            
        # Daha az sonuç döndür (daha hızlı)
        results = self.current_collection.query(
            query_texts=[user_query],
            include=["documents", "metadatas", "distances"],
            n_results=n_results
        )
        return results["documents"][0] if only_text else results

    def clear_current_pdf(self):
        """Sadece kullanıcı istediğinde PDF'i temizle"""
        if self.current_pdf_path:
            # Önbellekten kaldır
            pdf_hash = self._get_file_hash(self.current_pdf_path)
            if pdf_hash in self.cached_chunks:
                del self.cached_chunks[pdf_hash]
            
            # Collection'ı temizle
            if self.current_collection:
                try:
                    self.client.delete_collection(self.current_collection.name)
                except:
                    pass
            
            self.current_pdf_path = None
            self.current_collection = None
            

def format_answer_with_clickable_links(raw_answer):
    if "Sources:" in raw_answer:
        body, sources_raw = raw_answer.split("Sources:")
        links = [line.strip("-• ") for line in sources_raw.strip().splitlines() if line.strip()]
        html_links = "<br>".join([f'<a href="{url}" target="_blank">{url}</a>' for url in links])
        html_answer = f"<div style='font-family: sans-serif; line-height: 1.6'>{body.strip()}<br><br><b>Kaynaklar:</b><br>{html_links}</div>"
    else:
        html_answer = raw_answer
    return html_answer
rag_instance = FinSentioRAG(
                collection_name="FinSentioDocs",
                chroma_path=CHROMADB_PATH
            )
# Ana fonksiyon - Gradio chatbot için kullanılacak
def compute_md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
def generate_financial_response(prompt, pdf_file=None, history=None, clear_pdf=False):
    global rag_instance
    try:
        if history is None:
            history = []
        chat_history = []
        for h in history:
            if len(h) == 2:
                chat_history.append({"role": "user", "parts": [h[0]]})
                chat_history.append({"role": "model", "parts": [h[1]]})

        # PDF temizleme isteği varsa
        if pdf_file is None and rag_instance.current_pdf_path:
            rag_instance.clear_current_pdf()

        if clear_pdf and rag_instance:
            rag_instance.clear_current_pdf()
            return "PDF temizlendi."
        blk = danger_agent.generate_response(prompt).strip()
        if blk != "safe":
            return blk
        intent = "file_analysis" if pdf_file else intent_classifier.generate_response(prompt).strip()
        print("Intent:", intent)    

        if intent == "file_analysis" and pdf_file is not None:
            temp_path = "temp_upload.pdf"
            try:
                # PDF işleme
                pdf_saved = False
                
                # Daha hızlı dosya kopyalama
                if hasattr(pdf_file, 'name') and os.path.exists(pdf_file.name):
                    _, ext = os.path.splitext(pdf_file.name)
                    if ext.lower() != '.pdf':
                        return f"⚠️ Lütfen PDF dosyası yükleyin. Yüklenen dosya formatı: {ext}"
                    
                    import shutil
                    shutil.copy2(pdf_file.name, temp_path)
                    pdf_saved = True
                
                if not pdf_saved:
                    if hasattr(pdf_file, 'read'):
                        with open(temp_path, 'wb') as f:
                            f.write(pdf_file.read())
                            pdf_saved = True
                    elif isinstance(pdf_file, bytes):
                        with open(temp_path, 'wb') as f:
                            f.write(pdf_file)
                            pdf_saved = True
                
                # PDF formatı kontrolü
                with open(temp_path, 'rb') as f:
                    if f.read(4) != b'%PDF':
                        os.remove(temp_path)
                        return "⚠️ Yüklenen dosya geçerli bir PDF formatında değil."
                temp_hash = rag_instance._get_file_hash(temp_path)
                new_hash = compute_md5(temp_path)
                # RAG işlemi - sadece yeni PDF yüklendiğinde işle
                if rag_instance.current_pdf_hash is None or rag_instance.current_pdf_hash != new_hash:
                # 4a) Önce varsa eski veriyi sil
                    rag_instance.clear_current_pdf()
                    # 4b) Yeni PDF'i yükle ve hash'i güncelle
                    success = rag_instance.load_pdf(temp_path, category="user_uploaded")
                    if not success:
                        return "⚠️ PDF işlenemedi."
                    rag_instance.current_pdf_hash = new_hash
                
                # Daha az metin kullan
                texts = rag_instance.convert_pdf_to_text(temp_path)
                if not texts:
                    return "⚠️ PDF dosyasından metin çıkarılamadı."
                
                # İlk 3000 karakteri kontrol et
                rel_pdf = relevance_agent.generate_response(texts[0][:3000]).strip()
                if rel_pdf != "relevant":
                    return rel_pdf

                # Daha az sonuç kullan
                context = "\n".join(rag_instance.query(prompt, n_results=10, only_text=True))
                
                # Gemini yanıtı
                rag_chat_agent = genai.GenerativeModel(
                    "gemini-2.0-flash",
                    system_instruction=role_rag,
                    safety_settings=safety_config
                )
                
                full_context_prompt = f"[DOCUMENT EXCERPTS]\n{context}\n\n[USER QUESTION]\n{prompt}"
                
                if chat_history:
                    chat = rag_chat_agent.start_chat(history=chat_history)
                    response = chat.send_message(full_context_prompt)
                else:
                    response = rag_chat_agent.generate_content(full_context_prompt)
                
                return format_answer_with_clickable_links(response.text)
                
            except Exception as e:
                return f"⚠️ PDF işlenirken bir hata oluştu: {str(e)}"

        # Web search yapılacaksa
        elif intent == "web_search":
            rel = relevance_agent.generate_response(prompt).strip()
            if rel != "relevant":
                return rel
            keywords = agent_keyword_generator.generate_response(prompt)
            
            results = search_google(keywords)
            
            parsed = parse_search_results(results)
            
            top_results = parsed[:30]
            summary_input = {
                "query": prompt,
                "results": [
                    {"title": t, "snippet": s, "link": l}
                    for t, s, l in top_results
                ]
            }
            full_summary_prompt = f"Search Query: {summary_input['query']}\nSearch Results: {json.dumps(summary_input['results'], ensure_ascii=False)}"
            
            summarizer = genai.GenerativeModel(
                "gemini-2.0-flash",
                system_instruction=role_summarize,  
                safety_settings=safety_config
            )
            
            if chat_history:
                chat = summarizer.start_chat(history=chat_history)
                summary = chat.send_message(full_summary_prompt)
            else:
                summary = summarizer.generate_content(full_summary_prompt)

            if summary.prompt_feedback and summary.prompt_feedback.block_reason:
                return "Üretilen cevap güvenlik filtresine takıldı."

            return format_answer_with_clickable_links(summary.text)

    except Exception as e:
        return f"[Error]: {str(e)}"

    return "Finansal bir soru sorabilir veya analiz için bir PDF yükleyebilirsiniz."

def clear_pdf_and_chat():
    return None, []  # PDF'i ve chat geçmişini temizle
