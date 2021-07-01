import argparse
from nltk import sent_tokenize, word_tokenize
from numpy import random
import os
import pycld2
from re import findall
from text_normalization import normalize


def markov_model(generator, zero_counts=False):
    """Creates a Markov model.

    Args:
        generator: A generator of lines in the text that the Markov model is based on.
        zero_counts: Optional; If zero_counts is True the Markov model will include the word
            combinations that are not in the text the Markov model is based on. Their counts
            will be 0.

    Returns:
        A dict that represents the generated Markov model with keys as windows (tuples of strings)
        and values as dicts with keys as words after the window (strings) and values as the number
        of times when the key word is after the window (integers).
    """
    model = {}
    window_and_last_word = []
    for line in generator:
        words = [word for sentence in
                 [["начало предложения"] + [word for word in word_tokenize(sentence, language=language)] +
                  ["конец предложения"] for sentence in sent_tokenize(line, language=language)]
                 for word in sentence]
        for last_word in words:
            if len(window_and_last_word) < M + 1:
                window_and_last_word.append(last_word)
            else:
                window = tuple(window_and_last_word[:-1:])
                if zero_counts:
                    # Adding all words from the text to model[window] as possible continuations of the window
                    # and setting the number of these continuations to 0
                    for word in list(set(words)):
                        if window in model.keys():
                            model[window][word] = 0
                        else:
                            model[window] = {word: 0}
                # Counting the number of times when last_word is after the window
                if window in model.keys():
                    if last_word in model[window].keys():
                        model[window][last_word] += 1
                    else:
                        model[window][last_word] = 1
                else:
                    model[window] = {last_word: 1}
    return model


def count_probabilities(model, laplace_smoothing=False):
    """Counts probabilities for a Markov model.

    Args:
        model: A Markov model as a dict with keys as windows (tuples of strings) and values
            as dicts with keys as words after the window (strings) and values as the number
            of times when the key word is after the window (integers).
        laplace_smoothing: Optional; If laplace_smoothing is True the function will use
            Laplace smoothing to count the probabilities.

    Returns:
        A dict that represents probabilities for the Markov model with keys as windows
        (tuples of strings) and values as dicts with keys as words after the window
        (strings) and values as the probability of the key word after the window
        (integers).
    """
    probabilities = {}
    for window in model.keys():
        probabilities[window] = {}
        for last_word in model[window].keys():
            if laplace_smoothing:
                # Laplace smoothing
                probability = (model[window][last_word] + 1) / \
                              (sum(model[window].values()) + len(
                                  model[window]))
            else:
                probability = model[window][last_word] / sum(model[window].values())
            probabilities[window][last_word] = probability

        if laplace_smoothing:
            # Renormalization
            prob_sum = sum(probabilities[window].values())
            for word in probabilities[window].keys():
                probabilities[window][word] /= prob_sum
    return probabilities


if __name__ == "__main__":
    # Source data
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

    # Creating a generator of lines in the source text
    source_text_lines_generator = (''.join(findall(r'[\w\s.?!,;:]', ''.join(findall(r'\D', line))))
                                   for source_filename in os.listdir(source_dir)
                                   for line in open(os.path.join(source_dir, source_filename), encoding='utf-8'))

    # Detecting the language of the source text
    MAX_CHARS_TO_DETECT = 10000
    chars_to_detect_language = []
    for source_filename in os.listdir(source_dir):
        with open(os.path.join(source_dir, source_filename), encoding='utf-8') as source:
            while len(chars_to_detect_language) < MAX_CHARS_TO_DETECT:
                character = source.read(1)
                if character != '':
                    chars_to_detect_language.append(character)
                else:
                    break
    text_to_detect_language = ''.join(chars_to_detect_language)
    language = pycld2.detect(text_to_detect_language)[2][0][0].lower()

    # Removing all non-alphabetic elements that are not punctuation marks from the text

    # The set of all the words in the text
    # (including start- and end-of-sentence marks and without non-alphabetic elements that are not punctuation marks)
    source_words = list(set([word for sentence in
                        [["начало предложения"] + [word for word in word_tokenize(sentence, language)] +
                         ["конец предложения"]
                         for sentence in sent_tokenize(' '.join(''.join(findall(r'[\w\s.?!,;:]',
                                                                                ''.join(findall(r'\D', line))))
                                                                for source_filename in os.listdir(source_dir)
                                                                for line in open(os.path.join(source_dir,
                                                                                              source_filename),
                                                                                 encoding='utf-8')), language)]
                        for word in sentence]))

    # The first Markov model
    markov_model_1 = markov_model(source_text_lines_generator)

    # The second Markov model
    normalized_words_generator = (' '.join([' '.join(sentence) for sentence in
                                            normalize(''.join(findall(r'[\w\s.?!,;:]', ''.join(findall(r'\D', line)))),
                                                      language=language)])
                                  for source_filename in os.listdir(source_dir)
                                  for line in open(os.path.join(source_dir, source_filename), encoding='utf-8'))

    markov_model_2 = markov_model(normalized_words_generator, zero_counts=True)

    # The probabilities for the first Markov model
    probabilities_1 = count_probabilities(markov_model_1)

    # The probabilities for the second Markov model
    probabilities_2 = count_probabilities(markov_model_2, laplace_smoothing=True)

    # Reading the text from the destination file
    with open(dest_filename, encoding='utf-8') as dest:
        # The number of the words in every line in dest
        dest_words_number = [len(word_tokenize(line, language=language)) for line in dest]

    # Concatenating the lines which contain the last M words from dest
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

        # Isolating the last M words from dest
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

    # Checking if the destination file is empty
    if not dest_words:
        generated_words = ["начало предложения"]
        i = -M
    else:
        generated_words = dest_words.copy()
        normalized_generated_words = normalized_dest_words.copy()
        i = 0

        # Generating the first M words
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

    # Generating the remaining words
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

    # Deleting the words that were added from dest_words from generated_words
    if dest_words:
        generated_words = generated_words[len(dest_words)::]

    # Making a text from generated_words
    generated_text = ' '.join([word for word in generated_words
                               if word != "начало предложения" and word != "конец предложения"])

    # Writing the generated text to the destination file
    if dest_words:
        with open(dest_filename, 'a', encoding='utf-8') as dest:
            dest.write(" " + generated_text)
    else:
        with open(dest_filename, 'a', encoding='utf-8') as dest:
            dest.write(generated_text)
