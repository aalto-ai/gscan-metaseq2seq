import csv
import os

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
import tqdm


def get_most_recent_version(experiment_dir):
    versions = os.listdir(os.path.join(experiment_dir, "lightning_logs"))
    versions = [
        v
        for v in versions
        if os.path.isdir(
            os.path.join(experiment_dir, "lightning_logs", v, "checkpoints")
        )
    ]
    return sorted(versions, key=lambda x: int(x.split("_")[1]))[-1]


class ConsoleLogger(Logger):
    @property
    def name(self):
        return "MyLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_experiment
    def log_hyperparams(self, params,):
        tqdm.write(f"{params}\n")

    @rank_zero_experiment
    def log_metrics(self, metrics, step):
        tqdm.write(f"{step} - {metrics}")

    @rank_zero_experiment
    def finalize(self, status):
        pass


class LoadableCSVLogger(CSVLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_start(self, trainer, pl_module):
        if self.metrics:
            trainer.callback_metrics = self.metrics[-1]

    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment is None:
            load_csv = True
        else:
            load_csv = False

        experiment = super().experiment

        if load_csv:
            try:
                with open(experiment.metrics_file_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    experiment.metrics = list(reader)
                    print(
                        f"Restored CSV ({len(experiment.metrics)} lines to step {experiment.metrics[-1]['step']}) logs from {experiment.metrics_file_path}"
                    )
            except IOError:
                print(f"No csv log files to restore")

        return experiment
