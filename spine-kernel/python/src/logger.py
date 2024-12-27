import sys
import logging
from logging import handlers, log
import os
import context
from context import log_dir


def addLoggingLevel(levelName, levelNum, methodName=None, className=None):
    if not methodName:
        methodName = levelName.lower()

    levelName = levelName.upper()

    if hasattr(logging, levelName):
        raise AttributeError("{} already defined in logging module".format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError("{} already defined in logging module".format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError("{} already defined in logger class".format(methodName))

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


# add extra logging level: trace
addLoggingLevel("TRACE", logging.DEBUG - 5)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


class InfoFilter(logging.Filter):
    def filter(self, rec):
        return rec.levelno in (logging.TRACE, logging.DEBUG, logging.INFO)


# create a logger
logger = logging.getLogger("Astraea Logger")
logger.setLevel(logging.TRACE)  # Log等级总开关

# create a file logger handler
logfile = os.path.abspath(os.path.join(log_dir, "astraea.log"))
# os.makedirs(os.path.dirname(logfile), exist_ok=True)
fh = logging.FileHandler(logfile, mode="a", delay=True, encoding="utf-8")
fh.setLevel(logging.NOTSET)

# create a console logger handler
stdout_h = logging.StreamHandler(sys.stdout)
# set according to LOG_LEVEL
stdout_h.setLevel(LOG_LEVEL)
stdout_h.addFilter(InfoFilter())
stderr_h = logging.StreamHandler()
stderr_h.setLevel(logging.WARNING)

# logger format
formatter = logging.Formatter(
    "[%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d] %(message)s"
)
fh.setFormatter(formatter)
stdout_h.setFormatter(formatter)
stderr_h.setFormatter(formatter)

# add handlers to logger
# logger.addHandler(fh)
logger.addHandler(stdout_h)
logger.addHandler(stderr_h)