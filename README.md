# PokerBotRe

## Dependencies

Install the required packages:

```powershell
pip install PokerNow undetected-chromedriver setuptools keras tensorflow
```
`setuptools` is needed for Python 3.12+ because `undetected-chromedriver` still imports `distutils`.
`keras` and `tensorflow` are needed for `EquityModel.py` to load `equity_net.keras`.

Optional ranging model support:

```powershell
pip install catboost
```

## Training Data Conversion

Raw hand-history data goes in `data/raw`. Convert it into ranging-model CSV rows with:

```powershell
.\.venv\Scripts\python.exe InterpretTrainingData.py
```

Outputs are written to `data/interpreted`. By default `keras_equity` is computed. For a faster debug conversion that leaves it as `nan`, run:

```powershell
.\.venv\Scripts\python.exe InterpretTrainingData.py --skip-keras-equity
```

Keras equity is predicted in batches. You can tune the batch size with:

```powershell
.\.venv\Scripts\python.exe InterpretTrainingData.py --keras-batch-size 8192
```

## Ranging Model Training

Train the CatBoost action-probability model from interpreted CSV rows:

```powershell
.\.venv\Scripts\python.exe TrainRangingModel.py
```

Outputs are written to `models/ranging_action_model.cbm` and `models/ranging_action_model_metadata.json`.
By default, direct hole-card identity/strength features are excluded to reduce overfitting. To train a hole-card-aware model, add:

```powershell
.\.venv\Scripts\python.exe TrainRangingModel.py --include-hole-card-features
```

For a quick smoke test:

```powershell
.\.venv\Scripts\python.exe TrainRangingModel.py --max-rows 5000 --iterations 50
```

For class-balanced training:

```powershell
.\.venv\Scripts\python.exe TrainRangingModel.py --balanced
```

## CatBoost Simulator

Play against three CatBoost bots using the trained ranging action model:

```powershell
.\.venv\Scripts\python.exe PlayCatBoostSimulator.py --hands 10
```

Human action keys:

```text
f fold
x check
c call
m min raise
h half-pot raise
t three-quarter-pot raise
p pot raise
a all in
```

## Configuration

Set your PokerNow hero/player name with an environment variable:

```powershell
$env:POKERBOT_HERO_NAME = 'your_player_name'
```
