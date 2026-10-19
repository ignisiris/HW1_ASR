import math

# ( ╥ ﹏ ╥ )( ╥ ﹏ ╥ )( ╥ ﹏ ╥ )( ╥ ﹏ ╥ )( ╥ ﹏ ╥ )


def logadd(first, second):
    maximum = max(first, second)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log1p(math.exp(min(first, second) - maximum))


class CTCPrefixBeamSearch:
    def __init__(self, text_encoder, beam_size=16):
        self.text_encoder = text_encoder
        self.beam_size = beam_size
        self.blank = text_encoder.char2ind[text_encoder.EMPTY_TOK]

    def _search(self, log_probs):
        frames = log_probs.detach().cpu().tolist()
        beams = {(): (0.0, -math.inf)}
        for frame in frames:
            next_beams = {}

            def add(prefix, blank=-math.inf, nonblank=-math.inf):
                previous_blank, previous_nonblank = next_beams.get(
                    prefix, (-math.inf, -math.inf)
                )
                next_beams[prefix] = (
                    logadd(previous_blank, blank),
                    logadd(previous_nonblank, nonblank),
                )

            for prefix, (blank_score, nonblank_score) in beams.items():
                total = logadd(blank_score, nonblank_score)
                add(prefix, blank=total + frame[0])
                for token, token_score in enumerate(frame):
                    if token == 0 or token_score == -math.inf:
                        continue
                    if prefix and token == prefix[-1]:
                        add(prefix, nonblank=nonblank_score + token_score)
                        extension_score = blank_score
                    else:
                        extension_score = total
                    if extension_score != -math.inf:
                        add(prefix + (token,), nonblank=extension_score + token_score)
            ranked = sorted(
                next_beams.items(),
                key=lambda item: logadd(*item[1]),
                reverse=True,
            )
            beams = dict(ranked[: self.beam_size])
        return beams

    def decode(self, log_probs):
        beams = self._search(log_probs)
        prefix = max(beams, key=lambda tokens: logadd(*beams[tokens]))
        return self.text_encoder.decode(prefix)

    def decode_batch(self, log_probs, log_probs_length):
        predictions = []
        for sample, length in zip(log_probs, log_probs_length):
            predictions.append(self.decode(sample[: int(length)]))
        return predictions
