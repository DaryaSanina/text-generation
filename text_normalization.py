import nltk
import pycld2
from pymorphy2 import MorphAnalyzer
from re import findall
from sys import argv


def normalize(text, language):
    # удаление неинтересных слов из текста
    text = ''.join(findall(r'[\w\s.?!,;:]', text))

    # разбиение текста на предложения
    sentences = nltk.sent_tokenize(text, language=language)

    # разбиение предложений на слова
    words = [nltk.word_tokenize(sentence, language=language) for sentence in sentences]

    # стемминг текста на английском языке
    if language == 'english':
        stemmer = nltk.stem.PorterStemmer()
        words = [[stemmer.stem(word) for word in sentence] for sentence in words]

    # лемматизация текста на русском языке
    if language == 'russian':
        morph = MorphAnalyzer()
        words = [[morph.parse(word)[0].normal_form for word in sentence] for sentence in words]

    return words


if __name__ == '__main__':
    # входные данные
    source_filename = argv[1]
    dest_filename = argv[2]

    # считывание текста
    with open(source_filename, encoding='utf-8') as source:
        source_text = source.read()

    # определение языка текста
    source_text_language = pycld2.detect(source_text)[2][0][0].lower()

    # нормализация текста
    dest_text = '\n'.join([' '.join(sentence) for sentence in normalize(source_text, source_text_language)])

    # запись обработанного текста
    with open(dest_filename, 'w', encoding='utf-8') as dest:
        dest.write(dest_text)
