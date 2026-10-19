import pickle

from src.text_encoder.ctc_beam_search import CTCPrefixBeamSearch, logadd


class CTCBeamSearchLM(CTCPrefixBeamSearch):
    def __init__(
        self, text_encoder, lm_path, beam_size=32, lm_alpha=1.0, word_bonus=1.0
    ):
        super().__init__(text_encoder, beam_size=beam_size)
        with open(lm_path, "rb") as source:
            self.lm = pickle.load(source)
        self.lm_alpha = lm_alpha
        self.word_bonus = word_bonus

    def decode(self, log_probs):
        beams = self._search(log_probs)
        best_text = ""
        best_score = -float("inf")
        for tokens, masses in beams.items():
            text = self.text_encoder.decode(tokens)
            acoustic = logadd(*masses)
            language, _ = self.lm.sentence_score(text)
            score = (
                acoustic
                + self.lm_alpha * language
                + self.word_bonus * len(text.split())
            )
            if score > best_score:
                best_score = score
                best_text = text
        return best_text
