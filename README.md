# IPL predictor

Predicts the first-innings score, each team's chance of winning when batting or bowling first,
and what a team should choose if it wins the toss.

## Run it
    pip install -r requirements.txt
    python ipl_predictor.py train --data ipl_comprehensive_dataset.csv   # trains, prints a backtest, saves ipl_model.joblib
    streamlit run app.py                                                 # the interactive page

    python ipl_predictor.py predict --team1 "Mumbai Indians" --team2 "Chennai Super Kings" --venue wankhede

## How it works
- `team1`/`team2` in the CSV is only a listing order, so who batted first is derived from the toss.
- Team names and venue names are merged (Delhi Daredevils = Delhi Capitals, Feroz Shah Kotla = Arun Jaitley Stadium, ...).
- Features are built match by match in date order using only earlier matches (Elo rating, batting-first and
  chasing records, venue scoring level and batting-first win rate), so nothing leaks from the future.
- Score model: Ridge regression on how far an innings lands above or below the current league average.
- Win model: heavily regularised logistic regression.
- Backtest: trained on 2007-2023, tested on 2024-2026.

Retrain whenever you add new matches to the CSV.
