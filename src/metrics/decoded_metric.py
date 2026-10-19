from src.metrics.base_metric import BaseMetric
from src.text_encoder.ctc_beam_search import CTCPrefixBeamSearch
# (  ; ω ; )(  ; ω ; )(  ; ω ; )

class BeamMetric(BaseMetric):
    def __init__(self, text_encoder, metric_fn, name=None, beam_size=16):
        super().__init__(name=name)
        self.text_encoder = text_encoder
        self.decoder = CTCPrefixBeamSearch(text_encoder, beam_size=beam_size)
        self.metric_fn = metric_fn

    def __call__(self, log_probs, log_probs_length, text, **kwargs):
        predictions = self.decoder.decode_batch(log_probs, log_probs_length)
        values = []
        for target, prediction in zip(text, predictions):
            target = self.text_encoder.normalize_text(target)
            values.append(self.metric_fn(target, prediction))
        return sum(values) / len(values)
