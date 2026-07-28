# FoodLens — Təqdimat və Sual-Cavab Hazırlığı

Bu sənəd təqdimatdan əvvəl oxumaq üçündür. Hər bölmə: **nə etdik**, **necə**,
**niyə** (imtahanda ən vacib hissə) və **mümkün suallar + cavablar**.

---

## 0. Bir cümlə ilə layihə

> FoodLens yemək şəklindən: (1) yeməyin növünü **CNN** ilə tapır, (2) porsiya
> ölçüsünü **Computer Vision** ilə qiymətləndirir, (3) kalori/makro hesablayır,
> (4) **NLP/RAG** ilə fərdi məsləhət verir.

Dörd AI komponenti qəsdən ayrıdır, çünki layihənin tələbi **CV + CNN + NLP**-nin
görünən, real istifadəsidir — hazır API-yə "şəkil göndər, cavab al" yox.

---

## 1. Ümumi axın (pipeline)

`src/pipeline.py` → `analyze()` funksiyası bütün mərhələləri birləşdirir:

```
Şəkil (BGR)
  → check_quality()      # CV: bulanıqlıq / işıq / ölçü yoxlaması
  → detect_plate()       # CV: boşqabı tap (Hough dairələri)
  → segment_food()       # CV: GrabCut ilə yemək maskası
  → predict()            # CNN: EfficientNet-B0 → top-5 sinif + etimad
  → estimate_portion()   # CV: maska sahəsi → S/M/L → qram
  → macros_for()         # nutrition_db.json → kalori/zülal/karb/yağ
  → advise()             # NLP/RAG: məqsədə görə məsləhət
```

**Vacib nüans (imtahanda soruşa bilərlər):** seqmentasiya (maska) **yalnız porsiya
ölçüsü** üçün istifadə olunur, **təsnifat üçün yox**. Bunun səbəbini 3-cü bölmədə
izah edirəm.

---

## 2. CNN hissəsi (əsas model)

### Nə var
İki model müqayisə olunur (`src/cnn/models.py`, `train.py`, `evaluate.py`):

| Model | Nədir | Top-1 dəqiqlik | Parametr |
|---|---|---|---|
| **SimpleCNN** | Sıfırdan yazılmış kiçik CNN | ~50% | ~0.4M |
| **EfficientNet-B0** | ImageNet-də öyrədilmiş, transfer learning | **92.3%** | ~4M |

Dataset: **Food-101**-in 25 sinifli alt-çoxluğu. Hər sinifdə ~750 öyrətmə şəkli.

### Niyə iki model?
Bu, layihənin "elmi" hissəsidir: **sıfırdan öyrətmə vs transfer learning**
müqayisəsi. Nəticə göstərir ki, kiçik datasetdə (25×750 şəkil) transfer learning
sıfırdan öyrətməni kəskin üstələyir (92% vs 50%). Bu, real AI mühəndisliyinin əsas
dərsidir.

### Niyə EfficientNet-B0?
- CPU-da sürətli inference (~22 ms/şəkil).
- ImageNet ön-öyrətməsi → az data ilə yüksək dəqiqlik.
- Kiçik ölçü (~16 MB checkpoint), laptopda işləyir.

### Öyrətmə strategiyası (train.py)
- **İki fazalı transfer learning:** əvvəl yalnız yeni "baş" (classifier) öyrədilir
  (backbone dondurulmuş, 3 epox), sonra bütün şəbəkə kiçik lr ilə (lr/10).
- **AdamW + label smoothing 0.1 + cosine LR** cədvəli.
- **Early stopping** — val macro-F1 3 epox yaxşılaşmasa dayanır.
- **seed=42** hər yerdə → təkrar-istehsal (reproducibility).

### Mümkün suallar
- **"Niyə accuracy yox, macro-F1?"** → Siniflər bir az disbalanslıdır; macro-F1
  hər sinifə bərabər çəki verir, ona görə azlıq siniflərdə zəifliyi gizlətmir.
- **"Overfitting-in qarşısını necə aldın?"** → data augmentation (RandAugment,
  flip, color jitter), label smoothing, early stopping, transfer learning.
- **"Grad-CAM nədir?"** → Modelin **hara baxdığını** göstərən istilik xəritəsi
  (`gradcam.py`, hooks ilə əl ilə yazılıb). İzah oluna bilənlik (explainability)
  üçün — model pizza deyirsə, həqiqətən pizzaya baxıbmı görürük.

---

## 3. Bu sessiyada düzəldilən 1-ci problem: Seqmentasiya train/test uyğunsuzluğu

### Problem
Real şəkildə burger atanda "suşi" deyirdi. Səbəb **model deyildi**.

Pipeline şəkli modelə verməzdən əvvəl GrabCut ilə **fonu silib yeməyi kəsirdi**.
Amma model **tam şəkillər** üzərində öyrədilib. Yəni:

- Öyrətmə vaxtı model gördü: **tam şəkil**
- İş vaxtı model gördü: **fon-silinmiş kəsik**

Bu, klassik **train/test mismatch** (distribution shift) problemidir → dəqiqlik çökür.

### Sübut (ölçdüm)
| Giriş | hamburger nəticəsi |
|---|---|
| Tam şəkil (xam predict) | hamburger ✅ (0.76–0.96) |
| Fon-silinmiş kəsik (köhnə pipeline) | caesar_salad / ice_cream ❌ |

### Həll (`src/pipeline.py`)
Təsnifatı **tam şəkil** üzərində etdim. Seqmentasiya yalnız porsiya ölçüsü və
maska görüntüsü üçün qaldı. Grad-CAM da tam şəklə keçdi (uyğunluq üçün).

```python
# Əvvəl (SƏHV):
top5 = predictor.predict(crop if crop.size else img_bgr, topk=5)
# İndi (DÜZGÜN):
top5 = predictor.predict(img_bgr, topk=5)
```

Nəticə: hamburger 1/4 → **3/4**, digər siniflər 4/4.

### Mümkün sual
- **"Train/test mismatch nədir?"** → Modelin öyrədildiyi data ilə real işlədiyi
  datanın paylanması fərqlidirsə, model pis işləyir. Burada həll: girişi öyrətmə
  ilə eyni formada saxlamaq.

---

## 4. Bu sessiyada 2-ci problem: Naməlum yemək (OOD) qoruyucusu

### Problem
Model yalnız **25 yemək** bilir. Ona plov/kabab/apple_pie atsan, o yenə də bu 25-dən
birini seçir — çünki softmax cəmi 1-ə bərabər olmalıdır. Nəticə: **əminliklə səhv cavab**.

### Araşdırma (öz ölçdüyüm data ilə)
- Tanıdığı yeməkdə softmax etimadı adətən **>0.9**.
- Tanımadığı yeməkdə **~0.3–0.45**-ə düşür, amma tam sıfıra yox.
- **Vacib dürüst nəticə:** softmax OOD-də hələ də "həddindən artıq özünə güvənir"
  (overconfident). Tək astana mükəmməl ayırıcı deyil.

### Həll (`config.py` + `schemas.py` + `pipeline.py`)
`CONFIDENCE_MIN = 0.50` astanası. Top-1 etimad bundan aşağıdırsa,
`low_confidence=True` qoyulur və UI:
- "Model əmin deyil" xəbərdarlığı göstərir,
- ən yaxın **3 ehtimalı** sadalayır,
- istifadəçiyə **düzgün sinfi əl ilə seçmək** imkanı verir.

Astananı **datadan** seçdim: 0.50-də real şəkillərin ~83%-i "əminliklə" keçir,
naməlum yeməklərin təxminən yarısı tutulur.

### Niyə bu, "səhv təkid etməkdən" yaxşıdır?
Dürüst AI sistemi bilmədiyini "bilmirəm" deməlidir. İmtahan komissiyası bunu
xüsusi qiymətləndirir (calibration / uncertainty).

### Mümkün suallar
- **"Niyə softmax OOD üçün pisdir?"** → Softmax həmişə cəmi 1 verir və çox vaxt
  yanlış sinfə də yüksək bal verə bilir (overconfidence). Yaxşı OOD üçün ayrıca
  metodlar var (energy score, temperature scaling), amma 25-sinifli qapalı modeldə
  tam həll yoxdur.
- **"Astananı niyə 0.50 seçdin?"** → In-class və OOD şəkillərdə etimad paylanmasını
  ölçüb, in-class 83% saxlanan, OOD-nin yarısını tutan nöqtəni seçdim.

---

## 5. Bu sessiyada 3-cü problem: Keyfiyyət yoxlaması çox sərt idi

### Problem
`check_quality()` şəkilləri "bulanıq"a görə rədd edirdi. Laplacian variansı astanası
**100** idi — real yemək şəkillərinin (xüsusən arxa fonu bokeh olanlar) **12%-ni**
səhvən rədd edirdi.

### Həll (`src/cv/quality.py`)
400 real Food-101 şəklində Laplacian variansını ölçdüm:
- Astana 100 → 11.8% rədd (çox)
- Astana 40 → 0.8% rədd, amma həqiqi bulanıq şəkillər hələ tutulur

Astananı **100 → 40** endirdim (data ilə əsaslandırılmış).

### Mümkün sual
- **"Bulanıqlığı necə ölçürsən?"** → Laplacian operatoru kənarları (edges) tapır;
  onun variansı aşağıdırsa, şəkildə kəskin kənar azdır → bulanıqdır. Bu, standart
  "variance of Laplacian" metodudur.

---

## 6. Bu sessiyada əlavə: NLP Çatbot (müəllimin təklifi)

### Nə etdik (`src/nlp/chatbot.py`)
Məqsədə görə (**arıqlama / saxlama / kütlə artırma**) qidalanma məsləhəti verən
söhbət botu. Streamlit-də "💬 Məsləhətçi" tab-ı.

### Bu, sadəcə "LLM-ə göndər" DEYİL — əsl RAG
**RAG = Retrieval-Augmented Generation.** Axın:

```
İstifadəçi sualı
  → retrieve()          # sualı embed et, təlimatlardan ən uyğun 4 mətn parçasını tap
  → cavab qur           # tapılan mətn + profil + məqsəd əsasında
  → mənbələri göstər    # hansı sənəddən götürüldüyünü göstərir
```

- **Embeddings:** `sentence-transformers` (all-MiniLM-L6-v2) — mətnləri vektora çevirir.
- **Axtarış:** cosine similarity + Azərbaycan dili üçün leksik uyğunluq (hibrid).
- **Baza:** `data/guidelines/*.md` (arıqlama, kütlə artırma, zülal, natrium və s.).
  Kütlə artırma sənədini bu sessiyada mən yazdım.

### Niyə template rejimi də var?
`src/nlp/llm.py` üç provider dəstəkləyir: **anthropic → local (flan-t5) → template**.
Default **template**-dir: API açarı olmadan, deterministik, offline işləyir. Belə ki:
- Təqdimat günü internet/açar olmasa belə sistem **çökmür**.
- Cavab yenə RAG ilə əsaslandırılıб (tapılan mətndən qurulur), sadəcə söz seçimi
  şablonludur. API açarı qoysan, cavab daha səlis olur.

### Məqsədə uyğunluq
- Arıqlama → kalori **defisiti**, yüksək zülal/lif.
- Kütlə artırma → kalori **profisiti**, zülal 1.6–2.2 q/kq, güc məşqi.
Profildəki `goal` sahəsi cavabın istiqamətini dəyişir. Bot həm də bugünkü qalan
kalorini bilir ("daha ~X kkal yeriniz var").

### Mümkün suallar
- **"RAG niyə lazımdır, birbaşa LLM niyə yox?"** → RAG cavabı **öz məlumat bazana**
  bağlayır → uydurma (hallucination) azalır, mənbə göstərilə bilir, model kiçik ola
  bilər. Layihə tələbi də "parser + retrieval" idi, təkcə LLM yox.
- **"Embedding nədir?"** → Mətnin mənasını təmsil edən ədəd vektoru; oxşar mənalı
  mətnlərin vektorları yaxın olur, ona görə cosine similarity ilə axtarış işləyir.

---

## 7. Bu sessiyada əlavə: Onboarding + kalori hədəfi

### Nə etdik (`app/streamlit_app.py`)
Saytı ilk açanda **cinsiyyət, çəki, boy** (+ yaş, aktivlik, məqsəd) soruşan
"xoş gəldin" formu. Cavablara görə gündəlik kalori hədəfi hesablanır.

### Düstur: Mifflin-St Jeor (`src/db.py`)
```
BMR = 10×çəki + 6.25×boy − 5×yaş + (kişi: +5 / qadın: −161)
TDEE = BMR × aktivlik_faktoru
Hədəf = TDEE + məqsəd_düzəlişi (arıqlama −500, kütlə +400)
```
Bu, tibbi olaraq qəbul edilmiş, standart BMR düsturudur.

### Texniki nüans (soruşa bilərlər)
`st.session_state` istifadə etdim ki, onboarding və yan panel **sinxron** qalsın —
onboarding-də girdiyin dəyər yan paneldə də görünür və redaktə oluna bilir.

---

## 8. Kod keyfiyyəti (mühəndislik tərəfi)

Bunları qeyd etsən, yaxşı təəssürat yaradar:
- **Type hints** hər funksiyada + **Google-style docstrings**.
- **ruff** (linter) + **mypy** (tip yoxlayıcı) hər dəyişiklikdən keçir.
- **Testlər** kodla eyni vaxtda yazılır (`tests/` — çatbot üçün 3 yeni test əlavə etdim).
- **Config mərkəzləşdirilib** (`config.py`) — heç bir yerdə hardcoded yol yoxdur.
- **Deterministik seed=42** hər yerdə → nəticələr təkrar-istehsal oluna bilir.

---

## 9. Layihənin zəif nöqtələri (əvvəlcədən de, güclü görünərsən)

İmtahanda "məhdudiyyətlər nədir?" soruşurlar. Dürüst cavablar:

1. **Yalnız 25 yemək** (Food-101 alt-çoxluğu). Azərbaycan yeməkləri (plov, dolma,
   kabab) datasetdə yoxdur, ona görə model onları bilmir. Genişləndirmə üçün 101
   sinifə keçid hazırdır (`COLAB_TRAINING.md`), amma yeni model öyrətmək lazımdır.
2. **OOD detection mükəmməl deyil** — softmax overconfidence səbəbindən naməlum
   yeməklərin yarısı hələ keçə bilir. Qismən həll: aşağı-etimad qoruyucusu.
3. **Porsiya qiymətləndirməsi təxminidir** — boşqab olmadan miqyas dəqiq deyil.
4. **Template rejimi şablonludur** — API açarı ilə cavab keyfiyyəti xeyli artır.

---

## 10. 60 saniyəlik "elevator pitch" (əzbərlə)

> "FoodLens yemək şəklindən kalori və fərdi məsləhət verən bir sistemdir. Dörd AI
> mərhələsi var: OpenCV ilə şəkil keyfiyyəti və porsiya, EfficientNet-B0 transfer
> learning ilə yeməyin təsnifatı (92% dəqiqlik), nutrition bazasından makrolar, və
> RAG əsaslı NLP çatbotu ilə məqsədə uyğun məsləhət. Modeli sıfırdan yazılmış CNN ilə
> müqayisə etdim və transfer learning-in kiçik datasetdə üstünlüyünü göstərdim.
> Sistem tamamilə laptop CPU-da, offline işləyir və bilmədiyi yeməkdə 'əmin deyiləm'
> deyir."

---

Uğurlar! Sual versələr, hər cavabın arxasında **"niyə"** dur — bu sənəddə hər qərarın
səbəbi var.
