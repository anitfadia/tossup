"""
IPL predictor: first-innings score, win chance (bat vs bowl first) and toss advice.

    python ipl_predictor.py train   --data ipl_comprehensive_dataset.csv
    python ipl_predictor.py predict --team1 "Mumbai Indians" --team2 "Chennai Super Kings" --venue wankhede
    streamlit run app.py
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "ipl_model.joblib"
HOLDOUT_FROM = "2024-01-01"       # everything before this trains, everything after is the backtest

# ---------------------------------------------------------------- cleaning ---
TEAM_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

# (keyword in lower-cased raw name, canonical venue). Order matters; first match wins.
VENUE_RULES = [
    ("chinnaswamy", "M Chinnaswamy Stadium, Bengaluru"),
    ("wankhede", "Wankhede Stadium, Mumbai"),
    ("eden gardens", "Eden Gardens, Kolkata"),
    ("kotla", "Arun Jaitley Stadium, Delhi"),
    ("arun jaitley", "Arun Jaitley Stadium, Delhi"),
    ("chidambaram", "MA Chidambaram Stadium, Chennai"),
    ("rajiv gandhi", "Rajiv Gandhi Stadium, Hyderabad"),
    ("sawai mansingh", "Sawai Mansingh Stadium, Jaipur"),
    ("narendra modi", "Narendra Modi Stadium, Ahmedabad"),
    ("sardar patel", "Narendra Modi Stadium, Ahmedabad"),
    ("bindra", "IS Bindra Stadium, Mohali"),
    ("punjab cricket association", "IS Bindra Stadium, Mohali"),
    ("yadavindra", "Mullanpur Stadium, New Chandigarh"),
    ("dubai", "Dubai International Stadium"),
    ("zayed", "Sheikh Zayed Stadium, Abu Dhabi"),
    ("sharjah", "Sharjah Cricket Stadium"),
    ("ekana", "Ekana Stadium, Lucknow"),
    ("maharashtra cricket", "MCA Stadium, Pune"),
    ("subrata roy", "MCA Stadium, Pune"),
    ("dy patil", "DY Patil Stadium, Navi Mumbai"),
    ("brabourne", "Brabourne Stadium, Mumbai"),
    ("himachal", "HPCA Stadium, Dharamsala"),
    ("rajasekhara", "ACA-VDCA Stadium, Visakhapatnam"),
    ("barsapara", "Barsapara Stadium, Guwahati"),
    ("holkar", "Holkar Stadium, Indore"),
    ("saurashtra", "Saurashtra Stadium, Rajkot"),
    ("shaheed veer narayan", "Shaheed Veer Narayan Singh Stadium, Raipur"),
    ("jsca", "JSCA Stadium, Ranchi"),
]


def canon_venue(name):
    low = str(name).lower()
    for key, canon in VENUE_RULES:
        if key in low:
            return canon
    return str(name)


def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["date", "match_number"]).reset_index(drop=True)
    for col in ("team1", "team2", "toss_winner", "winner"):
        df[col] = df[col].replace(TEAM_MAP)
    df["venue"] = df["venue"].map(canon_venue)

    # team1/team2 is just listing order, so work out who really batted first from the toss
    df["bat_first"] = np.where(
        df.toss_decision == "bat", df.toss_winner,
        np.where(df.toss_winner == df.team1, df.team2, df.team1))
    df["chase"] = np.where(df.bat_first == df.team1, df.team2, df.team1)
    df["fi_runs"] = np.where(df.bat_first == df.team1, df.team1_runs, df.team2_runs)
    df["no_result"] = df.result_type.eq("no result")
    return df


# ---------------------------------------------------------------- features ---
def _shrunk(vals, k, decay=0.95, last=40):
    """Recency-weighted mean pulled towards 0 when there is little history."""
    if not vals:
        return 0.0
    v = np.asarray(vals[-last:], dtype=float)
    w = decay ** np.arange(len(v) - 1, -1, -1)
    return float((w * v).sum() / (w.sum() + k))


def _rate(rec, prior, k):
    wins, n = rec
    return (wins + prior * k) / (n + k)


class Tracker:
    """
    Walks through matches in date order. features() only ever sees matches that were already
    played, so training rows never leak the future. After the last match it holds the current
    state of every team and venue, which is what the predictor uses.
    """

    def __init__(self, home):
        self.home = home                    # team -> its most-used venue
        self.league = 160.0                 # moving average of first-innings runs (captures the scoring trend)
        self.season = None
        self.elo, self.bat, self.bowl, self.venue = {}, {}, {}, {}
        self.defend, self.chase, self.ven_bf = {}, {}, {}

    def start_match(self, season):
        if season != self.season:
            if self.season is not None:     # squads change between seasons: pull ratings back towards 1500
                self.elo = {t: 1500 + 0.7 * (e - 1500) for t, e in self.elo.items()}
            self.season = season

    def feats(self, b, c, v):
        """Features for team b batting first against team c at venue v."""
        return {
            "league": self.league,
            "venue_off": _shrunk(self.venue.get(v, []), 5, 0.97, 60),   # how many runs above average this ground gives
            "bat_off": _shrunk(self.bat.get(b, []), 6),       # runs above league average when batting first
            "bowl_off": _shrunk(self.bowl.get(c, []), 6),     # runs above league average conceded when bowling first
            "home_b": int(self.home.get(b) == v),
            "home_c": int(self.home.get(c) == v),
            "elo_diff": self.elo.get(b, 1500.0) - self.elo.get(c, 1500.0),
            "def_b": _rate(self.defend.get(b, (0, 0)), 0.46, 8),      # batting-first win rate of b
            "chase_c": _rate(self.chase.get(c, (0, 0)), 0.54, 8),     # chasing win rate of c
            "venue_bf": _rate(self.ven_bf.get(v, (0, 0)), 0.46, 10),  # batting-first win rate at the venue
        }

    def update(self, r):
        if r.no_result:
            return
        res = r.fi_runs - self.league
        self.bat.setdefault(r.bat_first, []).append(res)
        self.bowl.setdefault(r.chase, []).append(res)
        self.venue.setdefault(r.venue, []).append(res)
        self.league += 0.05 * (r.fi_runs - self.league)
        if not isinstance(r.winner, str):       # tie: score counts, result does not
            return
        b, c = r.bat_first, r.chase
        b_won = int(r.winner == b)
        eb, ec = self.elo.get(b, 1500.0), self.elo.get(c, 1500.0)
        exp = 1 / (1 + 10 ** ((ec - eb) / 400))
        self.elo[b], self.elo[c] = eb + 20 * (b_won - exp), ec - 20 * (b_won - exp)
        for table, team, won in ((self.defend, b, b_won), (self.chase, c, 1 - b_won)):
            w, n = table.get(team, (0, 0))
            table[team] = (w + won, n + 1)
        w, n = self.ven_bf.get(r.venue, (0, 0))
        self.ven_bf[r.venue] = (w + b_won, n + 1)


def build_table(df):
    both = pd.concat([df[["team1", "venue"]].set_axis(["team", "venue"], axis=1),
                      df[["team2", "venue"]].set_axis(["team", "venue"], axis=1)])
    home = both.groupby("team").venue.agg(lambda s: s.value_counts().idxmax()).to_dict()

    tracker, rows = Tracker(home), []
    for r in df.itertuples():
        tracker.start_match(r.season)
        f = tracker.feats(r.bat_first, r.chase, r.venue)
        f.update(date=r.date, season=r.season, fi_runs=r.fi_runs, no_result=r.no_result,
                 bat_won=int(r.winner == r.bat_first) if isinstance(r.winner, str) else np.nan)
        rows.append(f)
        tracker.update(r)
    return tracker, pd.DataFrame(rows)


# ------------------------------------------------------------------ models ---
# Score: predict how far the innings lands above/below the *current* league average. Modelling the
# residual (not the raw score) stops the model being dragged back to older, lower-scoring seasons.
SCORE_F = ["venue_off", "bat_off", "bowl_off", "home_b", "home_c"]
WIN_F = ["elo_diff", "def_b", "chase_c", "venue_bf"]


def score_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=50))


def win_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=1000))


def evaluate(tab):
    """Train on matches before HOLDOUT_FROM, test on the rest. Returns metrics and an 80% score range."""
    cut = pd.Timestamp(HOLDOUT_FROM)
    metrics = {}

    sc = tab[~tab.no_result]
    tr, te = sc[sc.date < cut], sc[sc.date >= cut]
    m = score_model().fit(tr[SCORE_F], tr.fi_runs - tr.league)
    pred = te.league + m.predict(te[SCORE_F])
    metrics["score_mae"] = mean_absolute_error(te.fi_runs, pred)
    metrics["score_baseline_mae"] = mean_absolute_error(te.fi_runs, te.league)
    print(f"\nFIRST-INNINGS SCORE  (train {len(tr)} matches, backtest {len(te)} matches from {HOLDOUT_FROM[:4]})")
    print(f"  baseline (league average)  MAE {metrics['score_baseline_mae']:5.1f} runs")
    print(f"  this model                 MAE {metrics['score_mae']:5.1f} runs")
    resid = te.fi_runs.to_numpy() - pred.to_numpy()
    interval = (float(np.percentile(resid, 10)), float(np.percentile(resid, 90)))

    wn = tab[tab.bat_won.notna()]
    tr, te = wn[wn.date < cut], wn[wn.date >= cut]
    m = win_model().fit(tr[WIN_F], tr.bat_won)
    p = m.predict_proba(te[WIN_F])[:, 1]
    base = np.full(len(te), tr.bat_won.mean())
    print(f"\nWIN CHANCE  (train {len(tr)} matches, backtest {len(te)} matches)")
    for name, q in (("baseline (bat-first rate)", base), ("this model", p)):
        acc = accuracy_score(te.bat_won, q > .5)
        print(f"  {name:<26} acc {acc:.3f}  logloss {log_loss(te.bat_won, q):.3f}  brier {brier_score_loss(te.bat_won, q):.3f}")
    metrics["win_acc"] = accuracy_score(te.bat_won, p > .5)
    metrics["win_baseline_acc"] = accuracy_score(te.bat_won, base > .5)
    metrics["win_logloss"], metrics["win_baseline_logloss"] = log_loss(te.bat_won, p), log_loss(te.bat_won, base)
    return metrics, interval


def venue_profile(df, years=6):
    d = df[(~df.no_result) & (df.date.dt.year > df.date.dt.year.max() - years)]
    rows = {}
    for v, g in d.groupby("venue"):
        decided = g[g.winner.notna()]
        rows[v] = {
            "matches": len(g),
            "avg_first_innings": g.fi_runs.mean(),
            "bat_first_win_pct": 100 * (decided.winner == decided.bat_first).mean() if len(decided) else np.nan,
            "toss_field_pct": 100 * (g.toss_decision == "field").mean(),
        }
    return pd.DataFrame(rows).T


def train(data_path, out_path=MODEL_PATH):
    df = load_data(data_path)
    print(f"{len(df)} matches, {df.season.nunique()} seasons, {df.venue.nunique()} venues after cleaning")
    tracker, tab = build_table(df)
    metrics, interval = evaluate(tab)

    sc, wn = tab[~tab.no_result], tab[tab.bat_won.notna()]
    score = score_model().fit(sc[SCORE_F], sc.fi_runs - sc.league)
    win = win_model().fit(wn[WIN_F], wn.bat_won)

    latest = df[df.season == df.season.max()]
    recent = df[df.date.dt.year > df.date.dt.year.max() - 3]
    joblib.dump({
        "tracker": tracker, "score": score, "win": win, "interval": interval, "metrics": metrics,
        "teams": sorted(set(latest.team1) | set(latest.team2)),
        "venues": sorted(recent.venue.unique()),
        "profile": venue_profile(df),
        "data_through": str(df.date.max().date()),
    }, out_path)
    print(f"\nSaved {out_path} (data through {df.date.max().date()})")


# -------------------------------------------------------------- prediction ---
def predict(model, team_a, team_b, venue):
    tracker, (lo, hi) = model["tracker"], model["interval"]
    side = {}
    for bat, chase in ((team_a, team_b), (team_b, team_a)):
        f = pd.DataFrame([tracker.feats(bat, chase, venue)])
        runs = float(f["league"].iloc[0] + model["score"].predict(f[SCORE_F])[0])
        side[bat] = {"runs": runs, "low": runs + lo, "high": runs + hi,
                     "p_bat_win": float(model["win"].predict_proba(f[WIN_F])[0, 1])}

    win = {team_a: {"bat": side[team_a]["p_bat_win"], "bowl": 1 - side[team_b]["p_bat_win"]},
           team_b: {"bat": side[team_b]["p_bat_win"], "bowl": 1 - side[team_a]["p_bat_win"]}}
    advice = {}
    for t in (team_a, team_b):
        diff = win[t]["bat"] - win[t]["bowl"]
        advice[t] = {"choose": "bat" if diff >= 0 else "bowl", "gain": abs(diff),
                     "marginal": abs(diff) < 0.03}
    return {"score": side, "win": win, "toss": advice}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="ipl_comprehensive_dataset.csv")
    p = sub.add_parser("predict")
    p.add_argument("--team1", required=True)
    p.add_argument("--team2", required=True)
    p.add_argument("--venue", required=True, help="part of the venue name, e.g. wankhede")
    args = ap.parse_args()

    if args.cmd == "train":
        return train(args.data)

    model = joblib.load(MODEL_PATH)
    venue = next((v for v in model["venues"] if args.venue.lower() in v.lower()), None)
    if venue is None:
        raise SystemExit("Unknown venue. Options:\n  " + "\n  ".join(model["venues"]))
    a, b = args.team1, args.team2
    out = predict(model, a, b, venue)
    print(f"\n{a} vs {b} at {venue}")
    for t in (a, b):
        s = out["score"][t]
        print(f"  {t} bats first: ~{s['runs']:.0f} runs (80% range {s['low']:.0f}-{s['high']:.0f})")
    print()
    for t in (a, b):
        w, adv = out["win"][t], out["toss"][t]
        note = "marginal" if adv["marginal"] else f"+{adv['gain'] * 100:.0f} pts"
        print(f"  {t}: wins {w['bat'] * 100:.0f}% batting first, {w['bowl'] * 100:.0f}% bowling first"
              f"  -> if it wins the toss: {adv['choose']} ({note})")


if __name__ == "__main__":
    from ipl_predictor import main as _main    # re-import so saved objects belong to the ipl_predictor module
    _main()
