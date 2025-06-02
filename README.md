# Finansal Asistan Projesi

Bu proje, finansal analiz ve risk değerlendirmesi yapabilen, kullanıcılara finansal konularda yardımcı olan bir yapay zeka asistanıdır.

## Proje Hakkında

Bu proje, LLM (Large Language Model) tabanlı bir Finansal Danışman ve Eğitim Asistanı olarak geliştirilmiştir. Projede, modern yapay zeka teknikleri başarıyla uygulanmıştır:

- **Non-parametric Grounding Teknikleri:**
  - Function calling
  - Multi-agent mimarisi
  - RAG (Retrieval-Augmented Generation)
  - Prompt engineering

Bu teknikler sayesinde:
- Modelin güncel veriye erişimi sağlanmıştır
- Farklı "agent"ların iş bölümü yapması gerçekleştirilmiştir
- Prompt'ların dinamik olarak oluşturulması sağlanmıştır
- Finansal sorulara kaynak referanslı yanıt sunulması mümkün kılınmıştır

### Öne Çıkan Özellikler
- Risk değerlendirmelerinin tablo ve özet rapor formatında sunumu
- Gerçek zamanlı piyasa içgörüleri
- PDF tabanlı belge analizi
- Çift dil desteği (Türkçe-İngilizce)

## Özellikler

- Finansal analiz ve raporlama
- Risk değerlendirmesi
- Veri toplama ve analiz
- Kullanıcı yönetimi
- PDF dosya işleme
- ChromaDB ile veri depolama
- LLM maliyet hesaplama

## Gereksinimler

Projeyi çalıştırmak için aşağıdaki gereksinimlere ihtiyacınız vardır:

```bash
pip install -r requirements.txt
```

## Kurulum

1. Projeyi klonlayın:
```bash
git clone [proje-url]
```

2. Gerekli bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

3. Veritabanını oluşturun:
```bash
python database.py
```

## Kullanım

Ana uygulamayı başlatmak için:

```bash
python app.py
```

## Proje Yapısı

- `app.py`: Ana uygulama dosyası
- `financial_assistant.py`: Finansal asistan modülü
- `riskanalyzer.py`: Risk analizi modülü
- `dataretrieval.py`: Veri toplama ve işleme modülü
- `database.py`: Veritabanı işlemleri
- `llm_cost_calculator.py`: LLM maliyet hesaplama modülü
- `ChromaDBData/`: ChromaDB veri depolama dizini
- `chroma_db/`: ChromaDB yapılandırma dizini

## Veritabanı

Proje SQLite veritabanı kullanmaktadır. Veritabanı dosyası `users.db` olarak kaydedilir.
