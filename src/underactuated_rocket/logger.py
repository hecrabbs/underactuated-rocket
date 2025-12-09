import logging
import os
from logging.handlers import QueueHandler, QueueListener
from queue import Queue


class FileLogListener:
    def __init__(self, log_filename, logger: logging.Logger):
        this_path = os.path.dirname(os.path.abspath(__file__))
        log_dir = f"{this_path}/../.."
        log_path = f"{log_dir}/{log_filename}"

        # If log already exists move it to rocket.log_old
        # Anything in the _old file will be overwritten
        if os.path.isfile(f"{log_dir}/{log_filename}"):
            os.replace(log_path, log_path + "_old")

        log_queue = Queue()
        queue_handler = QueueHandler(log_queue)

        logger.addHandler(queue_handler)

        formatter = logging.Formatter("%(levelname)s: %(message)s")

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)

        self.queue_listener = QueueListener(log_queue, file_handler)

    def start(self):
        self.queue_listener.start()

    def stop(self):
        self.queue_listener.stop()
