import logging
from logging import handlers


class LoggerFactory(object):
    log = None

    @staticmethod
    def create_logger(log_file, log_level):
        """
        A private method that interacts with the python
        logging module
        """
        # set the logging format
        log_format = '[%(asctime)s - %(levelname)s:%(lineno)d - %(filename)s - %(funcName)s] - %(message)s'

        # Initialize the class variable with logger object
        log = logging.getLogger(log_file)
        handler = handlers.TimedRotatingFileHandler(filename='tcc.log', when='midnight')
        handler.setFormatter(logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S'))
        log.setLevel(logging.INFO)
        log.addHandler(handler)

        # set the logging level based on the user selection
        if log_level == "INFO":
            log.setLevel(logging.INFO)
        elif log_level == "ERROR":
            log.setLevel(logging.ERROR)
        elif log_level == "DEBUG":
            log.setLevel(logging.DEBUG)
        return log

    @staticmethod
    def get_logger(log_file, log_level):
        """
        A static method called by other modules to initialize logger in
        their own module
        """
        logger = LoggerFactory.create_logger(log_file, log_level)

        # return the logger object
        return logger
