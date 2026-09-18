# -*- coding: utf-8 -*-
"""global logger config
"""
import logging

class SpecificLogFilter(logging.Filter):
    def filter(self, record):
        return 'timestamp' in record.getMessage()


def logger_config(log_savepath,logging_name):
    '''logger config
    '''
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)

    file_handler = logging.FileHandler(log_savepath, encoding='UTF-8')
    file_handler.setLevel(logging.DEBUG)


    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger


