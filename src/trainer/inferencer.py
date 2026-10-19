from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.text_encoder.ctc_beam_lm import CTCBeamSearchLM
from src.text_encoder.ctc_beam_search import CTCPrefixBeamSearch
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            text_encoder (CTCTextEncoder): text encoder.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.text_encoder = text_encoder

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        # TODO change inference logic so it suits ASR assignment
        # and task pipeline

        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        # Some saving logic. This is an example
        # Use if you need to save predictions on disk

        batch_size = batch["logits"].shape[0]
        current_id = batch_idx * batch_size

        for i in range(batch_size):
            # clone because of
            # https://github.com/pytorch/pytorch/issues/1995
            logits = batch["logits"][i].clone()
            label = batch["labels"][i].clone()
            pred_label = logits.argmax(dim=-1)

            output_id = current_id + i

            output = {
                "pred_label": pred_label,
                "label": label,
            }

            if self.save_path is not None:
                # you can use safetensors or other lib here
                torch.save(output, self.save_path / part / f"output_{output_id}.pth")

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result()


class ASRInferencer(Inferencer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder_name = self.cfg_trainer.get("decoder", "greedy")
        if self.decoder_name not in ("greedy", "beam", "beam_lm"):
            raise ValueError(
                "Unknown decoder: "
                + self.decoder_name
                + " .... but, in general, there’s no need to box yourself in"
            )
        self.beam = None
        if self.decoder_name == "beam":
            self.beam = CTCPrefixBeamSearch(
                self.text_encoder, beam_size=self.cfg_trainer.get("beam_size", 16)
            )
        elif (
            self.decoder_name == "beam_lm"
        ):  # тот же beam с пересчётом кандидатов по LM
            self.beam = CTCBeamSearchLM(
                self.text_encoder,
                lm_path=self.cfg_trainer.lm_path,  # файл обученной LM из конфига
                beam_size=self.cfg_trainer.get("beam_size", 32),
                lm_alpha=self.cfg_trainer.get("lm_alpha", 1.0),
                word_bonus=self.cfg_trainer.get("word_bonus", 1.0),
            )
        if self.save_path is not None:
            self.save_path = Path(self.save_path)

    @torch.no_grad()
    def process_batch(self, batch_idx, batch, metrics, part):
        batch = self.move_batch_to_device(dict(batch))
        if self.batch_transforms is not None:
            batch = self.transform_batch(batch)
        batch.update(self.model(**batch))

        if self.decoder_name in ("beam", "beam_lm"):
            predictions = self.beam.decode_batch(
                batch["log_probs"], batch["log_probs_length"]
            )
        else:
            indices = batch["log_probs"].argmax(-1).cpu()
            predictions = [
                self.text_encoder.ctc_decode(row[: int(length)])
                for row, length in zip(indices, batch["log_probs_length"])
            ]
        batch["predictions"] = predictions

        for audio_path, prediction in zip(batch["audio_path"], predictions):
            identifier = Path(audio_path).stem
            self._seen_ids.add(identifier)
            if self.save_path is not None:
                output = self.save_path / part / (identifier + ".txt")
                output.write_text(prediction + "\n", encoding="utf-8")

        labeled = [
            index for index, text in enumerate(batch["text"]) if text is not None
        ]
        self._labeled_count += len(labeled)
        if metrics is not None and labeled:
            for metric in self.metrics["inference"]:
                ___errors, count = metric(
                    log_probs=batch["log_probs"][labeled],
                    log_probs_length=batch["log_probs_length"][labeled],
                    text=[batch["text"][index] for index in labeled],
                )
                metrics.update(metric.name, ___errors, count, True)
        return batch

    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()
        self._seen_ids = set()
        self._labeled_count = 0
        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        for batch_idx, batch in tqdm(
            enumerate(dataloader), desc=part, total=len(dataloader)
        ):
            self.process_batch(batch_idx, batch, self.evaluation_metrics, part)

        logs = {"predictions": len(self._seen_ids), "labeled": self._labeled_count}
        if self.evaluation_metrics is not None and self._labeled_count:
            logs.update(self.evaluation_metrics.result())
        return logs
