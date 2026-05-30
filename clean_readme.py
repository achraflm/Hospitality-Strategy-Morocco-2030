import re

with open('README.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace inline backticks with bold
# We want to match `text` but NOT match ```text
# Using a regex: (?<!`)`([^`\n]+)`(?!`)
text = re.sub(r'(?<!`)`([^`\n]+)`(?!`)', r'**\1**', text)

# Replace inline math $R^2 = 0.532$ with R2 = 0.532
# Replace inline math $11.86\%$ with 11.86%
text = text.replace('$', '')
text = text.replace('\\%', '%')
text = text.replace('R^2', 'R2')
text = text.replace('\\mathcal{N}', 'N')
text = text.replace('\\mu', 'moyenne')
text = text.replace('\\sigma', 'ecart-type')
text = text.replace('\\text{', '')
text = text.replace('}', '')

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(text)
