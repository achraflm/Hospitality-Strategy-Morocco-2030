# AutoResearch Automated Evaluation Report

## Target: Arrivals

### Model Rankings
| Model         | Validation   |        R2 |   MAPE (%) |   SMAPE (%) |
|:--------------|:-------------|----------:|-----------:|------------:|
| XGBoost       | Walk-Forward |  0.532258 |    11.8616 |     12.1296 |
| LSTM 2 Layers | Walk-Forward | -0.125894 |    19.4364 |     19.0624 |
| LSTM + CNN    | Walk-Forward | -0.125927 |    19.4369 |     19.0627 |
| LSTM          | Walk-Forward | -0.12606  |    19.4378 |     19.0634 |
| GRU           | Walk-Forward | -0.126077 |    19.4379 |     19.0634 |

### AutoResearch Insights
- **XGBoost (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **LSTM 2 Layers (Walk-Forward)**: Model is worse than baseline average. Temporal degradation or structural break (e.g. COVID) likely broke generalization. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **LSTM + CNN (Walk-Forward)**: Model is worse than baseline average. Temporal degradation or structural break (e.g. COVID) likely broke generalization. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **LSTM (Walk-Forward)**: Model is worse than baseline average. Temporal degradation or structural break (e.g. COVID) likely broke generalization. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **GRU (Walk-Forward)**: Model is worse than baseline average. Temporal degradation or structural break (e.g. COVID) likely broke generalization. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.

---
## Target: Nights

### Model Rankings
| Model         | Validation   |       R2 |   MAPE (%) |   SMAPE (%) |
|:--------------|:-------------|---------:|-----------:|------------:|
| XGBoost       | Walk-Forward | 0.489894 |    12.1079 |     12.254  |
| GRU           | Walk-Forward | 0.352457 |    14.3777 |     14.4179 |
| LSTM + CNN    | Walk-Forward | 0.352457 |    14.3777 |     14.4179 |
| LSTM          | Walk-Forward | 0.352457 |    14.3777 |     14.4179 |
| LSTM 2 Layers | Walk-Forward | 0.352456 |    14.3777 |     14.4179 |

### AutoResearch Insights
- **XGBoost (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **GRU (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **LSTM + CNN (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **LSTM (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **LSTM 2 Layers (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.

---
