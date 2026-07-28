# 101 sinif üzrə modeli öyrətmək (Colab GPU)

FoodLens artıq **bütün 101 Food-101 yeməyini** tanıya bilər — bunun üçün modeli
GPU-da yenidən öyrətmək lazımdır (bu laptopda yalnız CPU var, ona görə öyrətmə
Colab-da aparılır; CLAUDE.md buna icazə verir).

## Addımlar

1. **Dəyişiklikləri GitHub-a göndər** (101 sinifli `config.py` və `nutrition_db.json` Colab-a çatsın):
   ```bash
   git add -A && git commit -m "Expand to 101 classes" && git push
   ```

2. **Colab-da aç:** [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) faylını
   Google Colab-a yüklə (və ya GitHub-dan aç: *File → Open notebook → GitHub*).

3. **GPU seç:** Runtime → Change runtime type → **GPU (T4)**.

4. Notebook hüceyrələrini sıra ilə işlət. O:
   - Food-101-i endirir və 101 sinif üçün bölgüləri qurur,
   - EfficientNet-B0-ı öyrədir (~45–75 dəq),
   - `models/effnet_best.pt` faylını endirir.

5. **Endirilmiş `effnet_best.pt`-i** layihədə köhnə `models/effnet_best.pt` ilə əvəz et.

6. Streamlit-i yenidən başlat:
   ```bash
   streamlit run app/streamlit_app.py
   ```

Bundan sonra sayt istənilən Food-101 yeməyini (pizza-dan paxlava, sushi-dən
bibimbap-a qədər) tanıyacaq. Qeyd: **Azərbaycan yeməkləri** (plov, dolma, kabab)
Food-101-də yoxdur, ona görə tanınmayacaq — bunun üçün ayrıca data toplamaq lazımdır.

## Vacib qeydlər

- **Astana (`CONFIDENCE_MIN`)**: model əmin olmayanda sayt "əmin deyiləm" +
  ən yaxın 3 ehtimal göstərir, tək səhv cavab vermir.
- **Smoke rejimi** artıq əsl checkpoint-i pozmur — `--smoke` ayrıca
  `effnet_smoke.pt` faylına yazır.
- Öyrətmədən sonra `python -m src.cnn.evaluate --model effnet` `reports/metrics.json`-u
  101 sinif üçün yeniləyir.
