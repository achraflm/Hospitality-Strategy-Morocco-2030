# AutoResearch Automated Evaluation Report

## Target: Arrivals

### Model Rankings
| Model         | Validation   |        R2 |   MAPE (%) |   SMAPE (%) |
|:--------------|:-------------|----------:|-----------:|------------:|
| XGBoost       | Walk-Forward |  0.532258 |    11.8616 |     12.1296 |
| LSTM + CNN    | Walk-Forward |  0.179879 |    17.2667 |     17.0588 |
| LSTM 2 Layers | Walk-Forward |  0.163353 |    15.8427 |     16.3635 |
| LSTM          | Walk-Forward | -0.116424 |    19.3647 |     19.0065 |
| GRU           | Walk-Forward | -0.11648  |    19.3651 |     19.0088 |

### AutoResearch Insights
- **XGBoost (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **LSTM + CNN (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **LSTM 2 Layers (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **LSTM (Walk-Forward)**: Model is worse than baseline average. Temporal degradation or structural break (e.g. COVID) likely broke generalization. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **GRU (Walk-Forward)**: Model is worse than baseline average. Temporal degradation or structural break (e.g. COVID) likely broke generalization. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.

---
## Target: Nights

### Model Rankings
| Model         | Validation   |       R2 |   MAPE (%) |   SMAPE (%) |
|:--------------|:-------------|---------:|-----------:|------------:|
| XGBoost       | Walk-Forward | 0.489894 |    12.1079 |     12.254  |
| LSTM 2 Layers | Walk-Forward | 0.352521 |    14.3773 |     14.4174 |
| LSTM + CNN    | Walk-Forward | 0.352514 |    14.3776 |     14.4175 |
| LSTM          | Walk-Forward | 0.352467 |    14.3776 |     14.4178 |
| GRU           | Walk-Forward | 0.352466 |    14.3777 |     14.4178 |

### AutoResearch Insights
- **XGBoost (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **LSTM 2 Layers (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **LSTM + CNN (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **LSTM (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **GRU (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.

---
