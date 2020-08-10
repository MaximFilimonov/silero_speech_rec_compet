# Silero Speech Activity Classifier Pipeline

## 1. Requirements
```
tqdm
numpy
torch
plotly
pandas
librosa
xgboost
seaborn
torchaudio
sklearn.metrics
sklearn.model_selection
```
## Feature extraction
- Low-level params:
  - RMS level; 
  - spectral centroid; 
  - bandwidth; 
  - zero-crossing rate; 
  - spectral roll-off freq;
- MFCC 20 first

## Models:
- XGBoost 

## Results
metric - acc: 0.95336
