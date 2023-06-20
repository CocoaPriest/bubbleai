import logging

c_handler = logging.StreamHandler()
c_format = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(filename)s:%(lineno)d ==> %(message)s"
)
c_handler.setFormatter(c_format)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(c_handler)
