import joblib
import pandas as pd
import streamlit as st

from ipl_predictor import MODEL_PATH, predict

st.set_page_config(page_title="IPL Predictor", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error("No trained model found. Run:  python ipl_predictor.py train --data ipl_comprehensive_dataset.csv")
    st.stop()

st.title("IPL match predictor")
st.caption(f"Built from every IPL match in the dataset, through {model['data_through']}.")

teams, venues = model["teams"], model["venues"]
col1, col2 = st.columns(2)
team_a = col1.selectbox("Team A", teams, index=0)
team_b = col2.selectbox("Team B", [t for t in teams if t != team_a], index=0)
venue = st.selectbox("Venue", venues)
toss = st.radio("Who won the toss?", ["Not decided yet", team_a, team_b], horizontal=True)

out = predict(model, team_a, team_b, venue)

# ---- 1. first-innings score -------------------------------------------------
st.subheader("Expected first-innings score")
cols = st.columns(2)
for col, team in zip(cols, (team_a, team_b)):
    s = out["score"][team]
    col.metric(f"{team} batting first", f"{s['runs']:.0f} runs")
    col.caption(f"80% of the time: {s['low']:.0f} to {s['high']:.0f}")

# ---- 2. win chance ----------------------------------------------------------
st.subheader("Chance of winning")
table = pd.DataFrame(
    {team: {"If batting first": f"{w['bat'] * 100:.0f}%", "If bowling first": f"{w['bowl'] * 100:.0f}%"}
     for team, w in out["win"].items()}).T
st.table(table)

# ---- 3. toss decision -------------------------------------------------------
st.subheader("What if you win the toss?")


def toss_message(team):
    adv = out["toss"][team]
    action = "bat" if adv["choose"] == "bat" else "bowl"
    if adv["marginal"]:
        return f"**{team}**: the toss decision barely matters here (under 3 points either way)."
    return f"**{team}** should **{action} first**: about {adv['gain'] * 100:.0f} points better chance of winning."


if toss == "Not decided yet":
    for team in (team_a, team_b):
        st.write(toss_message(team))
else:
    st.success(toss_message(toss))

# ---- 4. venue history -------------------------------------------------------
profile = model["profile"]
if venue in profile.index:
    p = profile.loc[venue]
    with st.expander("What this ground has done recently"):
        st.write(f"- {int(p.matches)} matches in the last 6 seasons")
        st.write(f"- Average first-innings score: {p.avg_first_innings:.0f}")
        st.write(f"- Team batting first has won {p.bat_first_win_pct:.0f}% of the time")
        st.write(f"- Toss winners chose to bowl {p.toss_field_pct:.0f}% of the time")

m = model["metrics"]
with st.expander("How reliable is this?"):
    st.write(
        f"Backtest on the most recent seasons (the model never saw them): the score estimate is off by about "
        f"{m['score_mae']:.0f} runs on average (always guessing the league average: {m['score_baseline_mae']:.0f}). "
        f"Win chances picked the winner {m['win_acc'] * 100:.0f}% of the time, which is close to a coin flip. "
        "T20 results are very noisy and this dataset only has match-level numbers, so treat the percentages "
        "as gentle nudges based on venue and team history, not as strong predictions.")
