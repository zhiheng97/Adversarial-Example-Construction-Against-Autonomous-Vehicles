import logging
from logging import handlers
from enum import Enum

class LabelEnum(Enum):
    UNKNOWN = 0
    UNKOWN_MOVABLE = 1
    UNKNOWN_UNMOVABLE = 2
    CAR = 3
    VAN = 4
    TRUCK = 5
    BUS = 6
    CYCLIST = 7
    MOTORCYCLIST = 8
    TRICYCLIST = 9
    PEDESTRIAN = 10
    TRAFFICCONE = 11

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def collate_fn(batch):
    return tuple(zip(*batch))

class Logger(object):
    level_relations = {
    'debug':logging.DEBUG,
    'info':logging.INFO,
    'warning':logging.WARNING,
    'error':logging.ERROR,
    'crit':logging.CRITICAL
    }
 
    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str) 
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')

        th.setFormatter(format_str)
        self.logger.addHandler(sh) 
        self.logger.addHandler(th)