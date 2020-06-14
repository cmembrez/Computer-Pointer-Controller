import logging
from pathlib import Path


class LogHelper:
    """
    layers.log: is specific to each models' layers and the information come from the .get_perf_counts OpenVINO API
    benchmark.log: capture the models' load time, input/output preprocessing times and inference time
    main.log: refers to information and error messages in the code in general
    """
    def __init__ (self):
        # files names
        project_path = Path(__file__).parent.parent.resolve()
        self.main_path = str(project_path) + '/log/main.log'
        self.main_name = "main"
        self.benchmark_path = str(project_path) + '/log/benchmark.log'
        self.benchmark_name = "benchmark"
        self.layers_path = str(project_path) + '/log/layers.log'
        self.layers_name = "layers"

        # main
        if logging.getLogger(self.main_name).hasHandlers():
            self.main = logging.getLogger(self.main_name)
        else:
            self.main = self.setup_logger(self.main_name, self.main_path, level=logging.DEBUG)

        # benchmark
        if logging.getLogger(self.benchmark_name).hasHandlers():
            self.benchmark = logging.getLogger(self.benchmark_name)
        else:
            self.benchmark = self.setup_logger(self.benchmark_name, self.benchmark_path, level=logging.DEBUG)

        # layers
        if logging.getLogger(self.layers_name).hasHandlers():
            self.layers = logging.getLogger(self.layers_name)
        else:
            self.layers = self.setup_logger(self.layers_name, self.layers_path, level=logging.DEBUG)

    def setup_logger(self, name, log_file, level=logging.INFO):
        formatter = logging.Formatter('%(asctime)s;%(levelname)s;%(message)s;')

        handler = logging.FileHandler(log_file, mode='w')  # the log files are overwritten, and not append (mode='a')
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger