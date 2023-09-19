import nltk
import matplotlib.pyplot as plt
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Download stopwords and punkt if you haven't already
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Read Moby Dick from Gutenberg
moby_dick = gutenberg.raw('melville-moby_dick.txt')

# Tokenization
tokens = word_tokenize(moby_dick)

# Stopwords filtering
stop_words = set(stopwords.words("english"))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Parts-of-Speech (POS) tagging
pos_tags = nltk.pos_tag(filtered_tokens)

# POS frequency
pos_counts = Counter(tag for word, tag in pos_tags)
common_pos = pos_counts.most_common(5)

print("5 Most Common Parts of Speech:")
for pos, count in common_pos:
    print(f"{pos}: {count}")

# Lemmatization
lemmatizer = WordNetLemmatizer()
top_lemmas = [lemmatizer.lemmatize(word, pos) for word, pos in pos_tags[:20]]

print("\nTop 20 Lemmas:")
print(top_lemmas)

# Plotting frequency distribution
pos_distribution = FreqDist(tag for word, tag in pos_tags)
pos_distribution.plot()

plt.show()
