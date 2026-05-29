# AutoResearch Automated Evaluation Report

## Target: Arrivals

### Model Rankings
| Model   | Validation   |       R2 |   MAPE (%) |   SMAPE (%) |
|:--------|:-------------|---------:|-----------:|------------:|
| XGBoost | Walk-Forward | 0.876056 |     7.4049 |     7.70432 |
| LSTM    | Walk-Forward | 0.695754 |    11.0543 |    11.9629  |
| GRU     | Walk-Forward | 0.599528 |    13.3105 |    14.3081  |

### AutoResearch Insights
- **XGBoost (Walk-Forward)**: Excellent generalization. The model successfully captures both trend and seasonality without severe overfitting. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **LSTM (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Systematic bias detected: The model consistently overpredicts or underpredicts. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **GRU (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Systematic bias detected: The model consistently overpredicts or underpredicts. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.

---
## Target: Nights

### Model Rankings
| Model   | Validation   |         R2 |   MAPE (%) |   SMAPE (%) |
|:--------|:-------------|-----------:|-----------:|------------:|
| XGBoost | Walk-Forward |  0.486989  |    12.7517 |     12.8435 |
| LSTM    | Walk-Forward |  0.26224   |    15.3817 |     16.2591 |
| GRU     | Walk-Forward | -0.0792377 |    21.3142 |     23.2267 |

### AutoResearch Insights
- **XGBoost (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **LSTM (Walk-Forward)**: Moderate performance. The model captures some dynamics but struggles with high variance. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.
- **GRU (Walk-Forward)**: Model is worse than baseline average. Temporal degradation or structural break (e.g. COVID) likely broke generalization. Residuals are relatively centered, indicating no severe systemic bias. Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.

---
