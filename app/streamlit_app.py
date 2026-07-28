"""FoodLens Streamlit demo. All user-facing strings in Azerbaijani."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from src import db  # noqa: E402
from src.nlp.chatbot import answer as chat_answer  # noqa: E402
from src.nlp.meal_parser import parse_meal  # noqa: E402
from src.nlp.summarizer import daily_summary  # noqa: E402
from src.pipeline import analysis_extras, analyze, load_nutrition_db, macros_for  # noqa: E402
from src.schemas import UserProfile  # noqa: E402

st.set_page_config(page_title="FoodLens", page_icon="🍽️", layout="wide")

# ---------- styling ----------
st.markdown("""
<style>
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
    .stApp { background: #FDF8F3; }
    h1, h2, h3 { color: #4A2C1A !important; font-family: Georgia, serif; }
    section[data-testid="stSidebar"] { border-right: 1px solid #EAD9C9; }
    .fl-card {
        background: #FFFFFF; border-radius: 16px; padding: 1.1rem 1.3rem;
        border: 1px solid #EAD9C9; box-shadow: 0 2px 10px rgba(74,44,26,.06);
        margin-bottom: 1rem;
    }
    .fl-hero {
        background: linear-gradient(120deg, #C2571B 0%, #E08A3C 100%);
        color: #FFF8F0; border-radius: 18px; padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .fl-hero .big { font-size: 2.1rem; font-weight: 800; line-height: 1.1; }
    .fl-hero .sub { opacity: .92; font-size: 1.02rem; margin-top: .3rem; }
    .fl-kpi { font-size: 1.7rem; font-weight: 700; color: #C2571B; }
    .fl-label { color: #8A6D57; font-size: .78rem; text-transform: uppercase;
                letter-spacing: .06em; }
    .fl-chip { display: inline-block; background: rgba(255,255,255,.25);
               border: 1px solid rgba(255,255,255,.5); color: inherit;
               border-radius: 999px; padding: .12rem .7rem; margin: .12rem;
               font-size: .8rem; }
    .fl-chip-dark { display: inline-block; background: #F6E7D8; color: #7A4A21;
               border-radius: 999px; padding: .15rem .7rem; margin: .1rem;
               font-size: .8rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1.05rem; padding: .6rem 1rem; }
    div.stButton > button {
        background: #C2571B; color: white; border: none; border-radius: 10px;
        padding: .55rem 1.4rem; font-weight: 600;
    }
    div.stButton > button:hover { background: #A34614; color: white; }
    div[data-testid="stFileUploader"] section {
        border: 2px dashed #D9B896; border-radius: 14px; background: #FFF9F2;
    }
</style>
""", unsafe_allow_html=True)


def fit_panel(img: np.ndarray, height: int = 300) -> np.ndarray:
    """Resize (and letterbox) any image to a fixed display height."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    new_w = max(int(w * height / h), 1)
    return cv2.resize(img, (new_w, height))


def effnet_available() -> bool:
    return (settings.models_dir / "effnet_best.pt").exists()


@st.cache_resource
def nutrition_db() -> dict:
    return load_nutrition_db()


ACTIVITY_AZ = {"sedentary": "Oturaq", "light": "Yüngül", "moderate": "Orta",
               "active": "Aktiv", "very_active": "Çox aktiv"}
GOAL_AZ = {"lose": "Arıqlamaq", "maintain": "Saxlamaq", "gain": "Kütlə artırmaq"}
SEX_AZ = {"m": "Kişi", "f": "Qadın"}

# Profile defaults live in session_state so onboarding and the sidebar stay in sync.
_PROFILE_DEFAULTS = {"name": "Qonaq", "age": 30, "sex": "m", "height_cm": 175,
                     "weight_kg": 75, "activity": "light", "goal": "maintain"}
for _k, _v in _PROFILE_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# ---------- onboarding: ask sex/weight/height up front to set the kcal target ----------
if not st.session_state.get("onboarded", False):
    st.markdown(
        '<div class="fl-hero"><div class="big">FoodLens-ə xoş gəldiniz 🍽️</div>'
        '<div class="sub">Sizə uyğun gündəlik kalori hədəfini hesablamaq üçün '
        'bir neçə məlumat lazımdır.</div></div>', unsafe_allow_html=True)
    with st.form("onboarding"):
        c1, c2 = st.columns(2)
        o_name = c1.text_input("Ad", st.session_state["name"])
        o_sex = c2.selectbox("Cinsiyyət", ["m", "f"],
                             index=["m", "f"].index(st.session_state["sex"]),
                             format_func=lambda s: SEX_AZ[s])
        o_age = c1.number_input("Yaş", 10, 100, int(st.session_state["age"]))
        o_height = c2.number_input("Boy (sm)", 100, 230,
                                   int(st.session_state["height_cm"]))
        o_weight = c1.number_input("Çəki (kq)", 30, 250,
                                   int(st.session_state["weight_kg"]))
        acts = list(ACTIVITY_AZ)
        o_activity = c2.selectbox("Aktivlik səviyyəsi", acts,
                                  index=acts.index(st.session_state["activity"]),
                                  format_func=lambda a: ACTIVITY_AZ[a])
        goals = list(GOAL_AZ)
        o_goal = st.selectbox("Məqsədiniz", goals,
                              index=goals.index(st.session_state["goal"]),
                              format_func=lambda g: GOAL_AZ[g])
        submitted = st.form_submit_button("Başla →", use_container_width=True)
    if submitted:
        st.session_state.update(name=o_name, age=o_age, sex=o_sex,
                                height_cm=o_height, weight_kg=o_weight,
                                activity=o_activity, goal=o_goal, onboarded=True)
        st.rerun()
    _t = db.daily_kcal_target(st.session_state["age"], st.session_state["sex"],
                              st.session_state["height_cm"],
                              st.session_state["weight_kg"],
                              st.session_state["activity"], st.session_state["goal"])
    st.info(f"Hazırkı seçimlərlə təxmini gündəlik hədəf: **~{_t:.0f} kkal**. "
            "Dəyişib **Başla** düyməsini basın.")
    st.stop()

# ---------- sidebar: profile (editable after onboarding) ----------
with st.sidebar:
    st.title("🍽️ FoodLens")
    st.caption("Yemək fotosundan kaloriyə, oradan məsləhətə")
    st.subheader("Profil")
    name = st.text_input("Ad", key="name")
    col_a, col_b = st.columns(2)
    age = col_a.number_input("Yaş", 10, 100, key="age")
    sex = col_b.selectbox("Cins", ["m", "f"], key="sex",
                          format_func=lambda s: SEX_AZ[s])
    height_cm = col_a.number_input("Boy (sm)", 100, 230, key="height_cm")
    weight_kg = col_b.number_input("Çəki (kq)", 30, 250, key="weight_kg")
    activity = st.selectbox("Aktivlik", list(ACTIVITY_AZ), key="activity",
                            format_func=lambda a: ACTIVITY_AZ[a])
    goal = st.selectbox("Məqsəd", list(GOAL_AZ), key="goal",
                        format_func=lambda g: GOAL_AZ[g])
    target = db.daily_kcal_target(age, sex, height_cm, weight_kg, activity, goal)
    st.markdown(f'<div class="fl-card"><div class="fl-label">Gündəlik hədəf</div>'
                f'<div class="fl-kpi">{target:.0f} kkal</div></div>',
                unsafe_allow_html=True)

profile = UserProfile(name=name, age=age, sex=sex, height_cm=height_cm,
                      weight_kg=weight_kg, activity=activity, goal=goal,
                      daily_kcal_target=float(target))


def get_user_id() -> int:
    session = db.get_session()
    user = db.get_or_create_user(session, name, age, sex, height_cm, weight_kg,
                                 activity, goal)
    return user.id


def log_meal(food_class: str, grams: float, confidence: float, source: str) -> None:
    macros, _, _ = macros_for(food_class, grams)
    session = db.get_session()
    row = db.MealLog(user_id=get_user_id(), food_class=food_class,
                     confidence=confidence, grams=grams, kcal=macros.kcal,
                     protein_g=macros.protein_g, carb_g=macros.carb_g,
                     fat_g=macros.fat_g, sodium_mg=macros.sodium_mg, source=source)
    session.add(row)
    session.commit()


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📷 Şəkil analizi", "✍️ Mətnlə əlavə et", "📊 Gündəlik", "💬 Məsləhətçi",
     "🧠 Model"])

# ---------- tab 1: photo analysis ----------
with tab1:
    st.header("Şəkil analizi")
    if not effnet_available():
        st.warning("Model hələ öyrədilməyib. Əvvəlcə `.\\run.ps1 train-effnet` işlədin.")
    src_choice = st.radio("Mənbə", ["Fayl yüklə", "Kamera"], horizontal=True,
                          label_visibility="collapsed")
    file = (st.file_uploader("Yemək şəkli seçin",
                             type=["jpg", "jpeg", "png", "webp", "bmp"])
            if src_choice == "Fayl yüklə" else st.camera_input("Şəkil çəkin"))

    if file is not None and effnet_available():
        img = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st.error("Bu şəkil formatı oxuna bilmədi. JPG və ya PNG yükləyin.")
            st.stop()
        with st.spinner("Analiz gedir..."):
            meal = analyze(img, profile, with_advice=False)
        if not meal.ok:
            for r in meal.quality_reasons_az:
                st.error(r)
        else:
            with st.spinner("Vizual nəticələr hazırlanır..."):
                extras = analysis_extras(img)

            db_all = load_nutrition_db()
            class_names = list(db_all.keys())

            def az_name_of(cls: str) -> str:
                return db_all[cls]["az_name"]

            if meal.low_confidence:
                cand = " · ".join(f"{az_name_of(c)} {v:.0%}"
                                  for c, v in meal.top5[:3])
                st.warning(
                    "⚠️ Model bu yeməkdən **əmin deyil** — tanıdığı "
                    f"yeməklərdən kənarda ola bilər. Ən yaxın ehtimallar: {cand}. "
                    "Düzgün yeməyi aşağıdan seçə bilərsiniz.")

            _idx = (class_names.index(meal.food_class)
                    if meal.food_class in class_names else 0)
            chosen = st.selectbox("Yemək (model səhv təxmin edibsə dəyişin)",
                                  class_names, index=_idx, format_func=az_name_of)
            if chosen != meal.food_class:
                meal.food_class = chosen
                _, meal.tags, meal.az_name = macros_for(chosen, meal.grams)

            top2 = (f" · 2-ci ehtimal: {az_name_of(meal.top5[1][0])} "
                    f"{meal.top5[1][1]:.0%}" if len(meal.top5) > 1 else "")
            st.markdown(f"""<div class="fl-hero">
                <div class="fl-label" style="color:#FFE8D2">Nəticə</div>
                <div class="big">{meal.az_name}</div>
                <div class="sub">Əminlik: <b>{meal.confidence:.0%}</b>{top2}<br>
                Porsiya: <b>{meal.portion_bucket}</b> ·
                {'boşqab tapıldı, miqyas dəqiqdir' if meal.portion_confidence == 'high'
                 else 'boşqab tapılmadı, təxmini qiymət'}</div>
                <div style="margin-top:.5rem">{''.join(
                    f'<span class="fl-chip">{t}</span>' for t in meal.tags)}</div>
                </div>""", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.image(cv2.cvtColor(fit_panel(img), cv2.COLOR_BGR2RGB),
                     caption="Orijinal", use_container_width=True)
            c2.image(fit_panel(extras["mask"]), caption="Seqmentasiya maskası",
                     clamp=True, use_container_width=True)
            c3.image(cv2.cvtColor(fit_panel(extras["gradcam"]), cv2.COLOR_BGR2RGB),
                     caption="Grad-CAM (model hara baxır)", use_container_width=True)

            grams = st.slider("Porsiya (qram), lazımsa düzəldin", 20, 1500,
                              int(meal.grams), 10)
            macros, _, _ = macros_for(meal.food_class, grams)
            k1, k2, k3, k4 = st.columns(4)
            for col, label, val in ((k1, "Kalori", f"{macros.kcal:.0f} kkal"),
                                    (k2, "Zülal", f"{macros.protein_g:.0f} q"),
                                    (k3, "Karbohidrat", f"{macros.carb_g:.0f} q"),
                                    (k4, "Yağ", f"{macros.fat_g:.0f} q")):
                col.markdown(f'<div class="fl-card"><div class="fl-label">{label}</div>'
                             f'<div class="fl-kpi">{val}</div></div>',
                             unsafe_allow_html=True)

            b1, b2 = st.columns(2)
            if b1.button("💬 Məsləhət al", use_container_width=True):
                from src.nlp.advisor import advise

                meal.grams = float(grams)
                meal.macros = macros
                with st.spinner("Məsləhət hazırlanır..."):
                    st.info(advise(meal, profile))
            if b2.button("📒 Gündəliyə yaz", use_container_width=True):
                log_meal(meal.food_class, float(grams), meal.confidence, "photo")
                st.success("Yemək gündəliyə əlavə olundu.")

# ---------- tab 2: text entry ----------
with tab2:
    st.header("Mətnlə əlavə et")
    text = st.text_input("Nə yemisiniz?",
                         placeholder="məs: 2 dilim pitsa və bir stəkan kola")
    if text:
        entities = parse_meal(text)
        if not entities:
            st.info("Heç bir yemək tapılmadı.")
        db_json = nutrition_db()
        for i, e in enumerate(entities):
            with st.container():
                cols = st.columns([2, 1, 1, 1, 1])
                if e.in_db:
                    az = db_json[e.food]["az_name"]
                    cols[0].markdown(f'<span class="fl-chip-dark">✅ {az}</span>',
                                     unsafe_allow_html=True)
                    default_g = float(db_json[e.food]["typical_serving_g"]) * e.qty
                    grams = cols[1].number_input("qram", 10.0, 3000.0, default_g,
                                                 key=f"g{i}")
                    m, _, _ = macros_for(e.food, grams)
                    cols[2].metric("kkal", f"{m.kcal:.0f}")
                    cols[3].write(f"{e.qty:g} {e.unit or 'porsiya'}")
                    if cols[4].button("Yaz", key=f"log{i}"):
                        log_meal(e.food, grams, 1.0, "text")
                        st.success(f"{az} gündəliyə yazıldı.")
                else:
                    cols[0].markdown(f'<span class="fl-chip-dark">❓ {e.raw}</span>',
                                     unsafe_allow_html=True)
                    cols[1].caption("Bazada yoxdur, kalori hesablanmadı")

# ---------- tab 3: daily log ----------
with tab3:
    st.header("Gündəlik")
    session = db.get_session()
    logs = db.logs_for_date(session, get_user_id(), dt.date.today())
    if not logs:
        st.info("Bu gün üçün qeyd yoxdur. Şəkil və ya mətnlə yemək əlavə edin.")
    else:
        total_kcal = sum(r.kcal for r in logs)
        st.progress(min(total_kcal / target, 1.0),
                    text=f"{total_kcal:.0f} / {target:.0f} kkal")
        db_json = nutrition_db()
        for r in logs:
            az = db_json.get(r.food_class, {}).get("az_name", r.food_class)
            st.markdown(f'<div class="fl-card">🍴 <b>{az}</b>, {r.grams:.0f} q, '
                        f'{r.kcal:.0f} kkal <span class="fl-label">'
                        f'({r.ts:%H:%M}, {"şəkil" if r.source == "photo" else "mətn"})'
                        f'</span></div>', unsafe_allow_html=True)

        import matplotlib.pyplot as plt

        p = sum(r.protein_g for r in logs) * 4
        c = sum(r.carb_g for r in logs) * 4
        f = sum(r.fat_g for r in logs) * 9
        if p + c + f > 0:
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            ax.pie([p, c, f], labels=["Zülal", "Karbohidrat", "Yağ"],
                   colors=["#8093f1", "#f3c178", "#e07b39"], autopct="%1.0f%%",
                   wedgeprops={"width": 0.42})
            fig.patch.set_alpha(0)
            st.pyplot(fig, use_container_width=False)

        if st.button("📋 Günü yekunlaşdır"):
            with st.spinner("Xülasə hazırlanır..."):
                s = daily_summary(logs, profile, date=str(dt.date.today()))
            st.info(s.summary_az)
            st.success(s.plan_az)

# ---------- tab 4: nutrition chatbot ----------
with tab4:
    st.header("Qidalanma məsləhətçisi")
    goal_az = {"lose": "arıqlamaq", "maintain": "çəkini saxlamaq",
               "gain": "kütlə artırmaq"}[goal]
    st.caption(f"Məqsədiniz: **{goal_az}** · Gündəlik hədəf: {target:.0f} kkal. "
               "Sual verin, cavablar profilinizə və məqsədinizə görə "
               "təlimatlardan hazırlanır.")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    session = db.get_session()
    logs = db.logs_for_date(session, get_user_id(), dt.date.today())
    remaining = float(target) - sum(r.kcal for r in logs)

    examples = ("Necə arıqlaya bilərəm?" if goal == "lose"
                else "Kütlə artırmaq üçün nə yeməliyəm?" if goal == "gain"
                else "Sağlam qidalanma üçün nə məsləhət edərsiniz?")
    cols = st.columns(3)
    quick = [examples, "Zülalı necə artırım?", "Şəkəri necə azaldım?"]
    picked = None
    for col, q in zip(cols, quick):
        if col.button(q, use_container_width=True):
            picked = q

    for role, text in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(text)

    msg = st.chat_input("Sualınızı yazın...") or picked
    if msg:
        st.session_state.chat.append(("user", msg))
        with st.chat_message("user"):
            st.markdown(msg)
        with st.chat_message("assistant"):
            with st.spinner("Düşünürəm..."):
                reply = chat_answer(msg, profile,
                                    history=st.session_state.chat[:-1],
                                    remaining_kcal=remaining)
            st.markdown(reply.text_az)
            if reply.sources:
                srcs = ", ".join(s.replace(".md", "") for s in reply.sources
                                 if s.endswith(".md"))
                if srcs:
                    st.caption(f"Mənbələr: {srcs}")
        st.session_state.chat.append(("assistant", reply.text_az))

    if st.session_state.chat and st.button("🗑️ Söhbəti təmizlə"):
        st.session_state.chat = []
        st.rerun()

# ---------- tab 5: model metrics ----------
with tab5:
    st.header("Model nəticələri")
    metrics_path = settings.reports_dir / "metrics.json"
    if not metrics_path.exists():
        st.info("Hələ qiymətləndirmə aparılmayıb: `.\\run.ps1 evaluate`")
    else:
        import json

        with metrics_path.open(encoding="utf-8") as fjson:
            metrics = json.load(fjson)
        rows = [{"Model": ("SimpleCNN (sıfırdan)" if k == "simple"
                           else "EfficientNet-B0 (transfer)"),
                 "Top-1": f"{v['top1_acc']:.1%}", "Top-5": f"{v['top5_acc']:.1%}",
                 "Macro F1": f"{v['macro_f1']:.3f}",
                 "Parametrlər": f"{v['params']:,}",
                 "CPU ms/şəkil": v["cpu_ms_per_image"]}
                for k, v in metrics.items()]
        st.table(rows)
        for img_name, caption in [
                ("model_comparison.png", "Model müqayisəsi"),
                ("confusion_matrix_effnet.png", "Confusion matrix (EfficientNet)"),
                ("confusion_matrix_simple.png", "Confusion matrix (SimpleCNN)")]:
            path = settings.reports_dir / img_name
            if path.exists():
                st.image(str(path), caption=caption)
        gradcam_dir = settings.reports_dir / "gradcam"
        if gradcam_dir.exists():
            st.subheader("Grad-CAM nümunələri")
            imgs = sorted(gradcam_dir.glob("*.png"))[:8]
            for row_start in range(0, len(imgs), 4):
                for col, p in zip(st.columns(4), imgs[row_start:row_start + 4]):
                    col.image(str(p), caption=p.stem[:40], use_container_width=True)
