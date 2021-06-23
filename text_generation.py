import argparse
from nltk import sent_tokenize, word_tokenize
from numpy import random
import os
import pycld2
from re import findall
from text_normalization import normalize


# создание марковской модели
def markov_model(words, zero_counts=False):
    model = {}
    for first_word_index in range(len(words) - M):
        window = tuple(words[first_word_index:first_word_index + M:])
        if window in model.keys():
            if words[first_word_index + M] in model[window].keys():
                model[window][words[first_word_index + M]] += 1
            else:
                model[window][words[first_word_index + M]] = 1
        else:
            if zero_counts:
                model[window] = {}
                for word_index in list(set(source_words)):
                    model[window][word_index] = 0
                model[window][source_words[first_word_index + M]] = 1
            else:
                model[window] = {words[first_word_index + M]: 1}
    return model


# вычисление вероятностей для марковской модели
def count_probabilities(model, laplace_smoothing=False):
    probabilities = {}
    for window in model.keys():
        probabilities[window] = {}
        for last_word in model[window].keys():
            if laplace_smoothing:
                # сглаживание Лапласа
                probability = (model[window][last_word] + 1) / \
                              (sum(model[window].values()) + len(
                                  model[window]))
            else:
                probability = model[window][last_word] / sum(model[window].values())
            probabilities[window][last_word] = probability

        if laplace_smoothing:
            # перенормировка
            prob_sum = sum(probabilities[window].values())
            for word in probabilities[window].keys():
                probabilities[window][word] /= prob_sum
    return probabilities


if __name__ == "__main__":
    # входные данные
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir', type=str)
    parser.add_argument('dest_filename', type=str)
    parser.add_argument('M', type=int, help="Order of the Markov model")
    parser.add_argument('N', type=int, help="Number of words to generate")
    args = parser.parse_args()

    source_dir = args.source_dir
    dest_filename = args.dest_filename
    M = args.M
    N = args.N

    # считывание текста
    source_texts = []
    for source_filename in os.listdir(source_dir):
        with open(os.path.join(source_dir, source_filename), encoding='utf-8') as source:
            source_texts.append(' '.join([line for line in source]))
    source_text = ' '.join(source_texts)

    # определение языка текста
    language = pycld2.detect(source_text)[2][0][0].lower()

    # удаление из текста всех небуквенных элементов, которые не являются знаками препинания
    source_text = ''.join(findall(r'[\w\s.?!,;:]', ''.join(findall(r'\D', source_text))))

    # список всех слов в тексте (включая обозначения начала и конца предложения)
    source_words = [word for sentence in
                    [["начало предложения"] + [word for word in word_tokenize(sentence)] +
                     ["конец предложения"] for sentence in sent_tokenize(source_text)]
                    for word in sentence]

    # первая марковская модель
    markov_model_1 = markov_model(source_words)

    # вторая марковская модель
    normalized_words = [word for sentence in
                        [["начало предложения"] + [word for word in sentence] +
                         ["конец предложения"] for sentence in normalize(source_text, language)]
                        for word in sentence]

    markov_model_2 = markov_model(normalized_words, zero_counts=True)

    # вычисление вероятностей для первой марковской модели
    probabilities_1 = count_probabilities(markov_model_1)

    # вычисление вероятностей для второй марковской модели
    probabilities_2 = count_probabilities(markov_model_2, laplace_smoothing=True)

    # считывание текста из выходного файла
    with open(dest_filename, encoding='utf-8') as dest:
        # количество слов в каждой строке dest
        dest_words_number = [len(word_tokenize(line, language=language)) for line in dest]

    # объединение строк, в которых содержатся M последних слов dest
    with open(dest_filename, encoding='utf-8') as dest:
        dest_list = []
        dest_text = ""
        for count, line in enumerate(dest):
            if count < len(dest_words_number) - 1:
                if sum(dest_words_number[count::]) >= M > sum(dest_words_number[count + 1::]):
                    dest_list.append(line)
            else:
                if sum(dest_words_number[count::]) >= M:
                    dest_list.append(line)
        dest_text = ' '.join(dest_list)

        # вычленение M последних слов dest
        dest_words = word_tokenize(dest_text, language=language)[::-1][:M:]

    dest_words = [word for sentence in
                  [["начало предложения"] + [word for word in word_tokenize(sentence, language)] +
                   ["конец предложения"] for sentence in sent_tokenize(' '.join(dest_words), language)]
                  for word in sentence][-M::]
    dest_text = ' '.join(word for word in dest_words if word != "начало предложения" and word != "конец предложения")
    normalized_dest_words = [word for sentence in
                             [["начало предложения"] + [word for word in sentence] +
                              ["конец предложения"] for sentence in normalize(dest_text, language)]
                             for word in sentence]

    # проверка выходного файла на пустоту
    if not dest_words:
        generated_words = ["начало предложения"]
        i = -M
    else:
        generated_words = dest_words.copy()
        normalized_generated_words = normalized_dest_words.copy()
        i = 0

        # генерация первых M слов
        j = 0
        while j < M:
            if tuple(normalized_generated_words[-M::]) in probabilities_2.keys():
                generated_words.append(random.choice(list(probabilities_2[tuple(normalized_generated_words[-M::])]
                                                          .keys()), p=list(probabilities_2[
                                                                               tuple(normalized_generated_words[
                                                                                     -M::])].values())))
            else:
                generated_words.append(random.choice(source_words))
            if generated_words[-1] != "начало предложения" and \
                    generated_words[-1] != "конец предложения":
                j += 1
                if findall(r'\w', generated_words[-1]) == list(generated_words[-1]):
                    normalized_generated_words.append(normalize(generated_words[-1], language)[0][0])
                else:
                    normalized_generated_words.append(generated_words[-1])
            else:
                normalized_generated_words.append(generated_words[-1])

    # генерация оставшихся слов
    while i < N - M:
        if len(generated_words) < M:
            if tuple(generated_words[-len(generated_words)::]) in probabilities_1.keys():
                generated_words.append(random.choice(list(probabilities_1[tuple(generated_words[
                                                                                -len(generated_words)::])].keys()),
                                                     p=list(probabilities_1[tuple(generated_words[
                                                                                  -len(generated_words)::])].values())))
            else:
                generated_words.append(random.choice(source_words))
        else:
            if tuple(generated_words[-M::]) in probabilities_1.keys():
                generated_words.append(random.choice(list(probabilities_1[tuple(generated_words[-M::])].keys()),
                                                     p=list(probabilities_1[tuple(generated_words[-M::])].values())))
            else:
                generated_words.append(random.choice(source_words))
        if generated_words[-1] != "начало предложения" and generated_words[-1] != "конец предложения":
            i += 1

    # удаление из generated_words слов, добавленных из dest_words
    if dest_words:
        generated_words = generated_words[len(dest_words)::]

    # составление текста из generated_words
    generated_text = ' '.join([word for word in generated_words
                               if word != "начало предложения" and word != "конец предложения"])

    # запись сгенерированного текста в файл
    if dest_words:
        with open(dest_filename, 'a', encoding='utf-8') as dest:
            dest.write(" " + generated_text)
    else:
        with open(dest_filename, 'a', encoding='utf-8') as dest:
            dest.write(generated_text)
