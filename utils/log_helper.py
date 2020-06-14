import logging
from pathlib import Path

class LogHelper:
    def __init__ (self):
        # files names
        project_path = Path(__file__).parent.parent.resolve()
        self.main_path = str(project_path) + '/log/main.log'
        self.main_name = "main"
        self.benchmark_path = str(project_path) + '/log/benchmark.log'
        self.benchmark_name = "benchmark"

        if logging.getLogger(self.main_name).hasHandlers():
            self.main = logging.getLogger(self.main_name)
        else:
            self.main = self.setup_logger(self.main_name, self.main_path, level=logging.DEBUG)

        if logging.getLogger(self.benchmark_name).hasHandlers():
            self.benchmark = logging.getLogger(self.benchmark_name)
        else:
            self.benchmark = self.setup_logger(self.benchmark_name, self.benchmark_path, level=logging.DEBUG)

    def setup_logger(self, name, log_file, level=logging.INFO):
        formatter = logging.Formatter('%(asctime)s;%(levelname)s;%(message)s;')

        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger