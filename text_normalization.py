import nltk
import pycld2
from pymorphy2 import MorphAnalyzer
from re import findall
from sys import argv

# A PorterStemmer object to stem the text in English and a MorphAnalyzer object to lemmatize the text in Russian
stemmer = nltk.stem.PorterStemmer()
morph = MorphAnalyzer()


def normalize(text, language):
    """Normalizes a text.

    Removes all non-alphanumeric characters that are not punctuation marks
    from the text. Stems the text if it is in English or lemmatizes the text
    if it is in Russian.

    Args:
        text: A text to normalize as a string.
        language: The full lowercase language of the text to normalize name as a string.
            Example: 'english'

    Returns:
        The normalized text as a string.
    """
    # Removing not interesting words from the text
    text = ''.join(findall(r'[\w\s.?!,;:]', text))

    # Splitting the text into sentences
    sentences = nltk.sent_tokenize(text, language=language)

    # Splitting the sentences into words
    words = [nltk.word_tokenize(sentence, language=language) for sentence in sentences]

    # Stemming the text in English
    if language == 'english':
        global stemmer
        words = [[stemmer.stem(word) for word in sentence] for sentence in words]

    # Lemmatizing the text in Russian
    if language == 'russian':
        global morph
        words = [[morph.parse(word)[0].normal_form for word in sentence] for sentence in words]

    return words


if __name__ == '__main__':
    # Source data
    source_filename = argv[1]
    dest_filename = argv[2]

    # Reading the text from the source file
    with open(source_filename, encoding='utf-8') as source:
        source_text = source.read()

    # Detecting the language of the text
    source_text_language = pycld2.detect(source_text)[2][0][0].lower()

    # Normalizing the text
    dest_text = '\n'.join([' '.join(sentence) for sentence in normalize(source_text, source_text_language)])

    # Writing the normalized text to the destination file
    with open(dest_filename, 'w', encoding='utf-8') as dest:
        dest.write(dest_text)
