from nltk.tokenize import regexp_tokenize

filler_words = [
    "um", "uh", "like", "you know", "well", "so", "actually", "basically",
    "literally", "honestly", "i mean", "kind of", "sort of", "anyway", "right",
    "okay", "just", "totally", "absolutely", "really"
]

# Create a regular expression pattern to tokenize text while preserving multi-word phrases
pattern = "|".join(filler_words)  # Concatenate filler words with "|" to match any of them
pattern = fr"\b{pattern}\b"  # Add word boundaries to match whole words only. r to avoid Python reading \b as back space

def count_filler_words(text):
    # Tokenize the text using regexp_tokenize and the defined pattern
    tokens = regexp_tokenize(text.lower(), pattern)

    print(tokens)
    # Count occurrences of filler words
    filler_word_counts = {word: tokens.count(word) for word in filler_words}

    return filler_word_counts

example_text = "Your example text goes here. You know, like, um, well, basically, uh, I mean, right?"
filler_word_counts = count_filler_words(example_text)

# Print results
print("Filler Word Counts:")
for word, count in filler_word_counts.items():
    if count >= 3:
        print(word, ": ", count)
