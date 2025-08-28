#Build a simple NLP pipeline in Python using NLTK that demonstrates the major steps of Natural Language Processing:
#Lexical Analysis (Segmentation & Morphological Analysis)


#Syntactic Analysis (Parsing & Grammar)


#Semantic Analysis (Meaning Representation)


#Pragmatic Analysis (Context Understanding)


#Machine Translation (Simple English-to-other-language translation)

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk import CFG, ChartParser
import string

# Download required NLTK data
nltk_packages = [
    'punkt', 
    'averaged_perceptron_tagger',
    'wordnet'
]
for pkg in nltk_packages:
    try:
        nltk.download(pkg)
    except:
        pass  # Already downloaded

# The English sentence
text = "The cat chased the mouse. It was hungry."

print("Original Text:", text)

# 1. Lexical Analysis
print("\n1. LEXICAL ANALYSIS")

# Sentence segmentation
sentences = sent_tokenize(text)
print("Sentence Segmentation:", sentences)

# Process each sentence separately
for i, sentence in enumerate(sentences):
    print(f"\nProcessing Sentence {i+1}: '{sentence}'")
    
    # Word tokenization
    tokens = word_tokenize(sentence)
    print("Word Tokens:", tokens)

    # Morphological analysis - Stemming
    stemmer = PorterStemmer()
    stems = [stemmer.stem(w) for w in tokens]
    print("Stems:", stems)

    # POS tagging
    pos_tags = pos_tag(tokens)
    print("POS Tags:", pos_tags)

    # Remove punctuation for parsing
    tokens_no_punct = [t for t in tokens if t not in string.punctuation]
    
    # 2. Syntactic Analysis
    print("\n2. SYNTACTIC ANALYSIS")
    
    # Define grammar for this specific sentence
    if "chased" in tokens:
        # Grammar for first sentence: "The cat chased the mouse."
        grammar = CFG.fromstring("""
        S -> NP VP
        NP -> Det N
        VP -> V NP
        Det -> 'The' | 'the'
        N -> 'cat' | 'mouse'
        V -> 'chased'
        """)
    else:
        # Grammar for second sentence: "It was hungry."
        grammar = CFG.fromstring("""
        S -> NP VP
        NP -> 'It'
        VP -> V Adj
        V -> 'was'
        Adj -> 'hungry'
        """)
    
    parser = ChartParser(grammar)
    
    # Try to parse the sentence
    parsed = False
    print("Parse Tree:")
    for tree in parser.parse(tokens_no_punct):
        tree.pretty_print()
        parsed = True
    
    if not parsed:
        print("No parse tree found with the defined grammar.")

# 3. Semantic Analysis
print("\n3. SEMANTIC ANALYSIS")

# Process the full text for semantic analysis
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)

# Create a simple semantic representation
def extract_meaning(pos_tags):
    # Find subjects (nouns), verbs, and objects
    subjects = []
    verbs = []
    objects = []
    
    for word, pos in pos_tags:
        if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP']:  # Nouns and pronouns
            if not verbs:  # If no verbs found yet, it's a subject
                subjects.append(word)
            else:  # If verbs found, it's an object
                objects.append(word)
        elif pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:  # Verbs
            verbs.append(word)
    
    if subjects and verbs and objects:
        return f"{verbs[0]}({subjects[0]}, {objects[0]})"
    elif subjects and verbs:  # For sentences like "It was hungry"
        return f"{verbs[0]}({subjects[0]})"
    return "Meaning representation not found"

meaning = extract_meaning(pos_tags)
print("Semantic Representation:", meaning)

# 4. Pragmatic Analysis
print("\n4. PRAGMATIC ANALYSIS")

# Context understanding - pronoun resolution
context = {
    "The cat": "Fluffy the pet cat", 
    "the mouse": "Jerry the cartoon mouse",
    "It": "The cat"  # pronoun reference
}

# Simple pronoun resolution
resolved_text = text
for key, value in context.items():
    resolved_text = resolved_text.replace(key, value)
    
print("Contextual meaning:", resolved_text)

# 5. Machine Translation
print("\n5. MACHINE TRANSLATION")

# English to Spanish dictionary
eng_to_spa = {
    "The": "El", "the": "el", 
    "cat": "gato", "mouse": "ratón", 
    "chased": "persiguió", "was": "estaba",
    "hungry": "hambriento", "It": "Él"
}

# Translate each word
translation = [eng_to_spa.get(w, w) for w in tokens]
print("Word-by-word translation:", " ".join(translation))

# better translation
def simple_translate(tokens):
    translated = []
    for i, token in enumerate(tokens):
        if token in eng_to_spa:
            # Capitalization
            if token[0].isupper() and i > 0:
                translated.append(eng_to_spa[token].capitalize())
            else:
                translated.append(eng_to_spa[token])
        else:
            translated.append(token)
    return " ".join(translated)

print("Improved translation:", simple_translate(tokens))


