#Build a simple NLP pipeline in Python using spaCy that demonstrates the major steps of Natural Language Processing:
#Lexical Analysis (Segmentation & Morphological Analysis)
#Syntactic Analysis (POS tagging & Dependency Parsing)
#Semantic Analysis (Word similarity, synonyms via WordNet)
#Pragmatic Analysis (Named Entity Recognition & Context Understanding)
#Machine Translation (English → another language)


import spacy
from collections import defaultdict

# Ensure you have the required packages installed.
# pip install spacy
# python -m spacy download en_core_web_sm

# English sentence
text = "Apple is looking at buying a UK startup in Silicon Valley for $1 billion."

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Please download the spaCy model: python -m spacy download en_core_web_sm")
    exit()

doc = nlp(text)

print("Original Text:", text)

# 1. Lexical Analysis
print("\n=== 1. LEXICAL ANALYSIS ===")
print("Tokens:", [token.text for token in doc])
print("Lemmas:", [token.lemma_ for token in doc])
print("Morphology:", [str(token.morph) for token in doc])
print("Stop Words:", [token.text for token in doc if token.is_stop])

# 2. Syntactic Analysis
print("\n=== 2. SYNTACTIC ANALYSIS ===")
print("POS Tags:", [(token.text, token.pos_, spacy.explain(token.pos_)) for token in doc])
print("\nDependency Parse:")
for token in doc:
    print(f"{token.text:<12} --> {token.dep_:<10} --> {token.head.text:<10} (Head POS: {token.head.pos_})")

# 3. Semantic Analysis (using spaCy's functionality)
print("\n=== 3. SEMANTIC ANALYSIS ===")

# Word similarity
if doc.has_vector:
    print("Document has vector representation")
    
    # Find similar words within the document
    if len(doc) > 1:
        similarity_score = doc[0].similarity(doc[1])
        print(f"Similarity between '{doc[0].text}' and '{doc[1].text}': {similarity_score:.3f}")
else:
    print("No word vectors available in this model. Use en_core_web_md or en_core_web_lg for better similarity analysis.")

# Simple synonym list
synonym_dict = {
    "startup": ["new company", "emerging business", "venture", "new enterprise"],
    "buying": ["purchasing", "acquiring", "taking over"],
    "looking": ["considering", "exploring", "evaluating"]
}

word = "startup"
if word in synonym_dict:
    print(f"Synonyms for '{word}': {', '.join(synonym_dict[word])}")
else:
    print(f"No synonyms found for '{word}' in our dictionary")

# 4. Pragmatic Analysis
print("\n=== 4. PRAGMATIC ANALYSIS ===")
print("Named Entities:")
for ent in doc.ents:
    print(f"  {ent.text:<20} --> {ent.label_:<10} (Description: {spacy.explain(ent.label_)})")

# Enhanced context understanding
context_info = {
    "organizations": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
    "locations": [ent.text for ent in doc.ents if ent.label_ == "GPE" or ent.label_ == "LOC"],
    "money": [ent.text for ent in doc.ents if ent.label_ == "MONEY"],
    "dates": [ent.text for ent in doc.ents if ent.label_ == "DATE"]
}

print("\nContext Analysis:")
for category, items in context_info.items():
    if items:
        print(f"  - {category.capitalize()}: {', '.join(items)}")

# Determine the overall context
if context_info["organizations"] and context_info["money"]:
    print("  This text appears to be about a business acquisition.")
elif any(keyword in text.lower() for keyword in ["buying", "acquisition", "merger"]):
    print("  This text discusses a potential business transaction.")

# 5. Machine Translation
print("\n=== 5. MACHINE TRANSLATION ===")

# Dictionary-based translation
eng_to_spa = {
    "Apple": "Apple",
    "is": "está",
    "looking": "buscando",
    "at": "en",
    "buying": "comprar",
    "a": "una",
    "startup": "startup",
    "in": "en",
    "Silicon": "Silicon",
    "Valley": "Valle",
    "for": "por",
    "$1 billion": "mil millones de dólares",
    "UK": "Reino Unido",
    ".": "."
}

# Handle multi-word entities properly
translated_parts = []
i = 0
while i < len(doc):
    # Check if the current token starts a named entity
    entity_found = False
    for ent in doc.ents:
        if ent.start == i:
            if ent.text in eng_to_spa:
                translated_parts.append(eng_to_spa[ent.text])
            else:
                # Translate each word of the entity separately
                for j in range(ent.start, ent.end):
                    token_text = doc[j].text
                    translated_parts.append(eng_to_spa.get(token_text, token_text))
            i = ent.end
            entity_found = True
            break
    
    if not entity_found:
        token_text = doc[i].text
        translated_parts.append(eng_to_spa.get(token_text, token_text))
        i += 1

print("Spanish Translation:", " ".join(translated_parts))
