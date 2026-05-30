import re

with open('docs/modeling.rst', 'rb') as f:
    content = f.read()

try:
    text = content.decode('utf-8')
except Exception:
    text = content.decode('utf-8', errors='ignore')

# Remove the weird wide text appended
split_phrase = "ces architectures avancées.\n"
if split_phrase in text:
    text = text.split(split_phrase)[0] + split_phrase
else:
    # If not found, look for another phrase
    split_phrase_2 = "performances robustes offertes par ces architectures"
    if split_phrase_2 in text:
        text = text.split(split_phrase_2)[0] + split_phrase_2 + " avancées.\n\n"

new_appendix = """
Projections Finales vers 2030
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

text += new_appendix

with open('docs/modeling.rst', 'w', encoding='utf-8') as f:
    f.write(text)
