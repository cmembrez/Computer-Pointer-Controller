import logging


class LogHelper:
    def __init__ (self):
        # files names
        LOG_FILENAME = '../log/main.log'
        BENCHMARK_LOG_FILENAME = '../log/benchmark.log'

        if logging.getLogger("main").hasHandlers():
            self.main = logging.getLogger("main")
        else:
            self.main = self.setup_logger("main", LOG_FILENAME, level=logging.DEBUG)

        if logging.getLogger("benchmark").hasHandlers():
            self.benchmark = logging.getLogger("benchmark")
        else:
            self.benchmark = self.setup_logger("benchmark", BENCHMARK_LOG_FILENAME, level=logging.DEBUG)

    def setup_logger(self, name, log_file, level=logging.INFO):
        formatter = logging.Formatter('%(asctime)s;%(levelname)s;%(message)s;')

        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger