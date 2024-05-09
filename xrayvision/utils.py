import logging
from typing import Union, Optional


def get_logger(name: str, level: Optional[Union[int, str]] = logging.WARNING) -> logging.Logger:
    """
    Return a configured logger instance.

    Parameters
    ----------
    name : `str`
        Name of the logger
    level : `int` or level, optional
        Level of the logger e.g `logging.DEBUG`

    Returns
    -------
    `logging.Logger`
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)  # type: ignore
    handler = logging.StreamHandler()
    handler.setLevel(level)  # type: ignore
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(lineno)s: %(message)s',
                                  datefmt='%Y-%m-%dT%H:%M:%SZ')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
