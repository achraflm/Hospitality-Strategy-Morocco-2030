import re
import shutil
import os

# Copy LSTM images to figures folder
os.makedirs('figures', exist_ok=True)
shutil.copy(r'notebooks\figures\LSTM_plus_CNN_Arrivals_WF_Prediction.png', r'figures\LSTM_Arrivals_Prediction.png')
try:
    shutil.copy(r'notebooks\figures\LSTM_plus_CNN_Nights_WF_Prediction.png', r'figures\LSTM_Nights_Prediction.png')
except:
    pass

with open('docs/modeling.rst', 'r', encoding='utf-8') as f:
    text = f.read()

# We want to replace the Projections Finales vers 2030 section
# specifically replacing XGBoost images with LSTM images
old_section = """Projections Finales vers 2030
-----------------------------

Les graphiques ci-dessous presentent les previsions finales pour les Arrivees et les Nuitees touristiques marocaines jusqu'en decembre 2030, incluant la Coupe du Monde, selon notre meilleur modele.

.. figure:: ../figures/prediction_2030_arrivees.png
   :width: 100%
   :align: center
   :alt: Prediction des Arrivees jusqu'a 2030

.. figure:: ../figures/prediction_2030_nuites.png
   :width: 100%
   :align: center
   :alt: Prediction des Nuitees jusqu'a 2030
"""

new_section = """Projections et Modèle Final (LSTM)
----------------------------------

Après évaluation, les prévisions finales retenues sont basées sur notre modèle Deep Learning LSTM (avec Walk-Forward). Voici les résultats de prédiction pour les Arrivées.

.. figure:: ../figures/LSTM_Arrivals_Prediction.png
   :width: 100%
   :align: center
   :alt: Prediction des Arrivees par LSTM

.. figure:: ../figures/LSTM_Nights_Prediction.png
   :width: 100%
   :align: center
   :alt: Prediction des Nuitees par LSTM
"""

if old_section in text:
    text = text.replace(old_section, new_section)
else:
    # Use regex to find it
    pattern = r"Projections Finales vers 2030.*?Resultats Complets des Modeles"
    text = re.sub(pattern, new_section + "\n\nResultats Complets des Modeles", text, flags=re.DOTALL)

# Remove the "Comparaison Graphique des Modeles Alternatives" section since we are now promoting LSTM as the main model
alt_section_pattern = r"Comparaison Graphique des Modeles Alternatives.*?Impact de l'Inflation"
text = re.sub(alt_section_pattern, "Impact de l'Inflation", text, flags=re.DOTALL)

with open('docs/modeling.rst', 'w', encoding='utf-8') as f:
    f.write(text)
