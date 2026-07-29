# FoodLens — Kodun Sadə İzahı (fayl-fayl)

Bu sənəd `src/` içindəki hər faylı **"nə edir + niyə belə yazılıb"** formatında izah edir.
Texniki terminlər sadə dildə açıqlanıb. Təqdimata qədər təkrar oxu.

---

## Ümumi axın (əvvəlcə bunu anla)

```
Şəkil → CV (keyfiyyət, seqmentasiya, porsiya) → CNN (yeməyin adı) →
        nutrition_db (kalori) → NLP (məsləhət/çatbot)
```

Hər şeyi birləşdirən fayl: **`pipeline.py`**. Aşağıda hər hissəni ayrıca izah edirik.

---

# 🎯 1. CV — Computer Vision (`src/cv/`)

## `quality.py` — Şəkil keyfiyyəti qapısı
**Nə edir:** pis şəkli əvvəldən rədd edir ki, model boş yerə səhv etməsin.
- **Bulanıqlıq:** `cv2.Laplacian(gray).var()`. Laplacian kənarları tapır; varians
  kiçikdirsə kəskin kənar azdır → bulanıq. Hədd = 40.
- **İşıq:** HSV-nin V (parlaqlıq) kanalının ortalaması çox aşağı/yüksək olsa rədd edir.
- **Ölçü:** tərəf < 224px olsa rədd edir.
**Niyə:** "zibil girsə, zibil çıxar" — girişi əvvəldən yoxlamaq daha etibarlıdır.

## `segment.py` — Boşqab tapma + yemək maskası
**Nə edir:** yeməyi fondan ayırır.
- `detect_plate()` → `cv2.HoughCircles` ilə boşqabı (dairəni) tapır; tapmasa
  `cv2.Canny` + `cv2.findContours` ilə ən böyük konturu götürür.
- `segment_food()` → `cv2.grabCut` (GrabCut alqoritmi) ilə fonu yemekdən ayırır,
  `morphologyEx` ilə maskanı təmizləyir, `connectedComponentsWithStats` ilə ən
  böyük parçanı saxlayır.
**Niyə:** porsiya ölçüsünü hesablamaq üçün yeməyin neçə piksel yer tutduğunu bilmək lazımdır.
**Vacib:** maska yalnız **porsiya** üçündür, təsnifat üçün deyil (train/test mismatch səbəbi).

## `portion.py` — Maska → qram
**Nə edir:** yeməyin çəkisini (qram) hesablayır, iki üsulla:
1. **Kateqoriya:** `coverage = maska/boşqab` → S/M/L → `qram = tipik_porsiya × faktor`.
2. **Miqyaslı (boşqab varsa):** boşqab 26 sm qəbul edilir → piksel sm²-ə çevrilir →
   `qram = sahə_sm² × sıxlıq`.
**Niyə:** kalori hesablamaq üçün qram lazımdır. Tək kameradan porsiya təxminidir,
ona görə boşqab yoxdursa "təxmini" (low) qeyd olunur — dürüstlük.

---

# 🎯 2. CNN — Yeməyin tanınması (`src/cnn/`)

## `dataset.py` — Data hazırlığı
**Nə edir:** şəkilləri modelin gözlədiyi formaya salır.
- `train_transform()` → öyrətmə üçün: təsadüfi kəsmə, çevirmə, rəng dəyişikliyi
  (**data augmentation** — model eyni şəkli müxtəlif görsün, overfitting azalsın).
- `eval_transform()` → test/istifadə üçün: 224×224-ə sal, normallaşdır (dəyişiklik yox).
- `FoodSubset` → `splits.json`-dan hansı şəklin hansı sinif olduğunu oxuyur.
**Niyə:** model sabit ölçülü, normallaşdırılmış giriş istəyir.

## `models.py` — Modellərin tərifi (ƏN VACİB fayl)
**Nə edir:** iki CNN qurur.
- `SimpleCNN` → **sıfırdan** yazılmış: 4 blok `Conv2d → BatchNorm → ReLU → MaxPool`,
  sonra `Linear(256 → siniflər)`. Conv kənar/tekstura tapır, MaxPool ölçünü kiçildir.
- `build_effnet()` → **transfer learning:** `EfficientNet_B0_Weights.IMAGENET1K_V1`
  (ImageNet biliyi hazır gəlir), yalnız son qat dəyişdirilir:
  `model.classifier[1] = nn.Linear(1280, 101)` → 1280 əlamətdən 101 yemək ehtimalına.
**Niyə iki model:** sıfırdan (50%) vs transfer learning (92%) müqayisəsi — az data-da
transfer learning-in üstünlüyünü göstərmək (layihənin elmi hissəsi).

## `train.py` — Öyrətmə
**Nə edir:** modeli data ilə öyrədir.
- **İki fazalı:** əvvəl yalnız yeni baş öyrədilir (backbone dondurulmuş), sonra hamısı
  kiçik lr ilə (fine-tune).
- **AdamW** (optimallaşdırıcı) + **label smoothing** (özünə həddən artıq güvənməsin) +
  **cosine LR** (öyrənmə sürəti tədricən azalır).
- **Early stopping:** validation nəticəsi 3 epox yaxşılaşmasa dayanır (overfitting-ə qarşı).
- Ən yaxşı modeli `effnet_best.pt`-ə saxlayır (siniflər, dəqiqlik metadata ilə).
**Niyə:** `--smoke` (test) ayrıca `effnet_smoke.pt`-ə yazır ki, əsl modeli pozmasın.

## `evaluate.py` — Qiymətləndirmə
**Nə edir:** test dəstində modelin nə qədər yaxşı olduğunu ölçür.
- **top-1** (birinci təxmin düzdürmü), **top-5**, **macro-F1**, confusion matrix,
  per-class F1 → `reports/`-a çıxarır.
**Niyə macro-F1:** hər sinifə bərabər çəki verir, disbalansı gizlətmir.

## `gradcam.py` — İzahedilənlik
**Nə edir:** modelin şəkildə **hara baxdığını** istilik xəritəsi ilə göstərir.
- **Hooks** (qarmaqlar) ilə son conv qatının aktivasiyalarını və qradiyentlərini tutur,
  onları birləşdirib "vacib sahələr" xəritəsi çıxarır (`applyColorMap` ilə rəngləyir).
**Niyə:** model "pizza" deyəndə, həqiqətən pizzaya baxıbmı görürük — etibar + səhv analizi.

## `predict.py` — Tanıma (istifadə)
**Nə edir:** bir şəkli alıb yeməyin adını qaytarır.
```python
probs = torch.softmax(self.model(self.to_tensor(img)), dim=1)[0]  # 101 ehtimal
vals, idxs = probs.topk(5)                                        # ən yüksək 5
```
**Niyə:** softmax xam rəqəmləri ehtimala çevirir (cəmi=1), `.topk` ən güclü təxminləri verir.

---

# 🎯 3. NLP — Təbii dil (`src/nlp/`)

## `meal_parser.py` — Mətndən yemək tapma (NER)
**Nə edir:** "2 dilim pitsa və bir stəkan kola" → `[(pizza, 2, dilim), (cola, 1, stəkan)]`.
- **Rəqəm** ("iki"→2, "yarım"→0.5), **vahid** (dilim, boşqab...), **yemək adı** tapır.
- Əvvəl **lüğət** (sinonim cədvəli) ilə axtarır (Azərbaycan dili üçün etibarlı), tapmasa
  **MiniLM embedding** ilə oxşarlıq axtarır (hədd 0.75). Tapılmasa "unknown" qeyd edir.
**Niyə:** şəkil olmadan, mətnlə də yemək əlavə etmək üçün.

## `retriever.py` — RAG axtarışı (embedding)
**Nə edir:** sualı uyğun məlumat mətnləri ilə tapır.
- Təlimat sənədlərini (`guidelines/*.md`) + hər yeməyin qidalanma sənədini **vektora**
  çevirir (`sentence-transformers`, MiniLM), `models/rag_index.npz`-ə saxlayır.
- Sual gələndə: sualı vektora çevir → **cosine similarity** + Azərbaycan dili üçün
  leksik uyğunluq (hibrid) → ən uyğun 4 mətni qaytarır.
**Niyə:** RAG-ın "Retrieval" hissəsi — cavabı öz bazamıza bağlamaq, uydurmanı azaltmaq.

## `advisor.py` — Bir yemək üçün məsləhət
**Nə edir:** analiz olunan yemək + profil → 3-4 cümləlik məsləhət (kalori payı,
əvəzləmə təklifi, duz/şəkər xəbərdarlığı).
**Niyə:** template rejimində Jinja şablonla, LLM rejimində RAG konteksti ilə — hər ikisi
mənbəyə bağlı.

## `chatbot.py` — Söhbət botu (məqsədə görə)
**Nə edir:** istifadəçinin sualına, **məqsədinə** (arıqlama/kütlə artırma) uyğun cavab verir.
- Sualı `retrieve()` ilə təlimatlardan əsaslandırır, məqsədə görə intro qurur
  (defisit/profisit), bugünkü qalan kalorini nəzərə alır.
- Template rejimində offline işləyir, LLM rejimində daha səlis.
**Niyə:** müəllimin təklifi — ehtiyaca görə (kökəlmə/arıqlama) məsləhət.

## `summarizer.py` — Günün xülasəsi + sabahkı plan
**Nə edir:** günün yemək qeydlərini yığır → xülasə (kalori, makro, xəbərdarlıq) +
sabah üçün 3 yeməklik plan (məqsədə görə).
**Niyə:** gündəlik izləmə və planlaşdırma — real diyet tətbiqi kimi.

## `llm.py` — LLM provider abstraksiyası
**Nə edir:** mətn generasiyası üçün 3 provider: **anthropic → local (flan-t5) → template**.
Biri işləməsə, avtomatik növbətiyə keçir.
**Niyə:** default **template** — API açarı/internet olmadan, offline, deterministik işləsin
(təqdimat günü çökməsin). Açar qoysan, cavab keyfiyyəti artır.

---

# 🔗 4. Birləşdirici və köməkçi fayllar

## `pipeline.py` — Hər şeyi birləşdirir
**Nə edir:** `analyze(şəkil)` → keyfiyyət → seqmentasiya → **təsnifat (tam şəkil)** →
porsiya → kalori → məsləhət → tam nəticə (`MealAnalysis`).
**Vacib sətir:** `predictor.predict(img_bgr)` — təsnifat tam şəkil üzərində (mismatch fix).

## `schemas.py` — Data strukturları
**Nə edir:** `Macros`, `UserProfile`, `MealAnalysis`, `DailySummary` — modullar arasında
gəzən sadə data konteynerləri (dataclass). `low_confidence` sahəsi burada.
**Niyə:** hər yerdə eyni struktur → dolaşıqlıq olmasın.

## `db.py` — Verilənlər bazası (SQLite)
**Nə edir:** istifadəçi və yemək qeydlərini saxlayır (SQLAlchemy). `daily_kcal_target()` —
Mifflin-St Jeor düsturu ilə gündəlik kalori hədəfini hesablayır.
**Niyə:** yemək gündəliyi və fərdi hədəf üçün.

## `api.py` — FastAPI (köməkçi)
**Nə edir:** pipeline-ı HTTP endpoint kimi açır (proqram xarici müraciət edə bilsin).
**Niyə:** Streamlit UI-dan əlavə, proqramlaşdırma interfeysi göstərmək üçün.

---

## 🎤 Ən çox sual alan 3 fayl (bunları yaxşı bil)
1. **`models.py`** — transfer learning (EfficientNet + dəyişdirilmiş son qat)
2. **`predict.py`** — softmax + topk (tanıma məntiqi)
3. **`retriever.py` / `chatbot.py`** — RAG (embedding + axtarış)

Bu üçünü izah edə bilsən, CNN + NLP suallarının çoxunu bağlayırsan.
