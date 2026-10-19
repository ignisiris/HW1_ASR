import math
from collections import Counter, defaultdict

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"

FALLBACK_DISCOUNTS = (0.5, 1.0, 1.5)


def estimate_discounts(counts):
    count_frequencies = Counter(counts)
    n1 = count_frequencies[1]
    n2 = count_frequencies[2]
    denominator = n1 + 2 * n2
    correction = n1 / denominator if denominator else None
    discounts = []
    fallback_classes = []
    for count, fallback in enumerate(FALLBACK_DISCOUNTS, start=1):
        discount = None
        if correction is not None and count_frequencies[count]:
            discount = (
                count
                - (count + 1)
                * correction
                * count_frequencies[count + 1]
                / count_frequencies[count]
            )
        if discount is None or not 0 < discount <= count:
            discount = fallback
            fallback_classes.append(count)
        discounts.append(discount)
    diagnostics = {
        "count_of_counts": {count: count_frequencies[count] for count in range(1, 5)},
        "fallback_classes": fallback_classes,
    }
    return discounts, diagnostics


class ModifiedKneserNeyTrigram:
    def __init__(self, texts):
        self.training_sentences = 0
        self.training_words = 0
        unigrams, bigrams, trigrams = self._count_ngrams(texts)
        self.vocabulary = set(unigrams) | {UNK}
        self.vocabulary_size = len(self.vocabulary)
        self.discounts = {}
        self.discount_diagnostics = {}
        self.tables = {}
        self.backoffs = {}

        for order, counts in ((1, unigrams), (2, bigrams), (3, trigrams)):
            discounts, diagnostics = estimate_discounts(counts.values())
            self.discounts[order] = discounts
            self.discount_diagnostics[order] = diagnostics
            self.tables[order], self.backoffs[order] = self._build_tables(
                counts, order, discounts
            )

    def _count_ngrams(self, texts):
        trigrams = Counter()
        sentence_starts = Counter()
        for text in texts:
            words = text.split()
            if not words:
                continue
            self.training_sentences += 1
            self.training_words += len(words)
            tokens = [BOS, BOS] + words + [EOS]
            sentence_starts[words[0]] += 1
            trigrams.update(zip(tokens, tokens[1:], tokens[2:]))
        if not trigrams:
            raise ValueError("No training text found.")

        bigrams = Counter((second, third) for first, second, third in trigrams)
        for word, count in sentence_starts.items():
            bigrams[BOS, word] = count
        unigrams = Counter(word for previous, word in bigrams)
        return unigrams, bigrams, trigrams

    @staticmethod
    def _build_tables(counts, order, discounts):
        counts_by_history = defaultdict(dict)
        for ngram, count in counts.items():
            history = () if order == 1 else ngram[:-1]
            word = ngram if order == 1 else ngram[-1]
            counts_by_history[history][word] = count

        probabilities = {}
        backoffs = {}
        for history, word_counts in counts_by_history.items():
            context_count = sum(word_counts.values())
            probabilities[history] = {
                word: max(0, count - discounts[min(count, 3) - 1]) / context_count
                for word, count in word_counts.items()
            }
            backoffs[history] = (
                sum(discounts[min(count, 3) - 1] for count in word_counts.values())
                / context_count
            )
        return probabilities, backoffs

    def normalize_word(self, word):
        return word if word in self.vocabulary or word == BOS else UNK

    def probability(self, word, history=()):
        word = self.normalize_word(word)
        history = tuple(self.normalize_word(word) for word in history[-2:])
        probability = (
            self.tables[1][()].get(word, 0)
            + self.backoffs[1][()] / self.vocabulary_size
        )
        if history:
            context = history[-1:]
            probability = (
                self.tables[2].get(context, {}).get(word, 0)
                + self.backoffs[2].get(context, 1.0) * probability
            )
        if len(history) == 2:
            probability = (
                self.tables[3].get(history, {}).get(word, 0)
                + self.backoffs[3].get(history, 1.0) * probability
            )
        return probability

    def sentence_score(self, text):
        words = text.split()
        history = (BOS, BOS)
        score = 0.0
        unknown_words = 0
        for word in words + [EOS]:
            unknown_words += word not in self.vocabulary
            score += math.log(self.probability(word, history))
            history = (history[-1], self.normalize_word(word))
        return score, unknown_words
