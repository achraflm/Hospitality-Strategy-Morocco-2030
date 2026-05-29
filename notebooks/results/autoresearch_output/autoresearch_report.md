# AutoResearch Automated Evaluation Report

## Target: Arrivals

### Model Rankings
| Model         | Validation   |       R2 |   MAPE (%) |   SMAPE (%) |
|:--------------|:-------------|---------:|-----------:|------------:|
| LSTM + CNN    | Walk-Forward | 0.873345 |    7.69094 |     7.84736 |
| LSTM          | Walk-Forward | 0.864665 |    8.00539 |     7.95626 |
| LSTM 2 Layers | Walk-Forward | 0.860161 |    8.43432 |     8.87522 |
| GRU           | Walk-Forward | 0.784499 |    9.79753 |    11.0674  |

### AutoResearch Insights
- **LSTM + CNN (Walk-Forward)**: Excellent generalization. The model successfully captures both trend and seasonality without severe overfitting. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **LSTM (Walk-Forward)**: Excellent generalization. The model successfully captures both trend and seasonality without severe overfitting. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **LSTM 2 Layers (Walk-Forward)**: Excellent generalization. The model successfully captures both trend and seasonality without severe overfitting. Systematic bias detected: The model consistently overpredicts or underpredicts. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **GRU (Walk-Forward)**: Excellent generalization. The model successfully captures both trend and seasonality without severe overfitting. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.

---
## Target: Nights

### Model Rankings
| Model         | Validation   |        R2 |   MAPE (%) |   SMAPE (%) |
|:--------------|:-------------|----------:|-----------:|------------:|
| LSTM + CNN    | Walk-Forward |  0.491135 |    13.471  |     13.6798 |
| GRU           | Walk-Forward |  0.435785 |    11.7984 |     12.2131 |
| LSTM          | Walk-Forward |  0.359299 |    13.3554 |     14.4403 |
| LSTM 2 Layers | Walk-Forward | -0.118954 |    19.0568 |     22.6001 |

### AutoResearch Insights
- **LSTM + CNN (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **GRU (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **LSTM (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **LSTM 2 Layers (Walk-Forward)**: Model is worse than baseline average. Temporal degradation or structural break (e.g. COVID) likely broke generalization. Systematic bias detected: The model consistently overpredicts or underpredicts. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.

---
