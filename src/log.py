# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import logging
class FCFLLog:
    def __init__(self, file="/var/log/fcfl.log", level=logging.INFO):
        self.level = level
        self.filename = file
        #logging.getLogger(__name__).findCaller(stack_info=False, stacklevel=3)
        logging.basicConfig(filename=self.filename,
                filemode = 'a',
                format="%(asctime)s %(funcName)s: %(message)s",
                level=self.level)

    def setLevel(self, level):
        self.level= level

    def getLevel(self):
        return self.level

    def log(self, msg):
        logging.log(self.level, msg)

    def critical(self, msg):
        logging.log(logging.CRITICAL, msg)

    def error(self, msg):
        logging.log(logging.ERROR, msg)

    def warning(self,msg):
        logging.log(logging.WARNING, msg)

    def info(self, msg):
        logging.log(logging.INFO, msg)
        print(msg)

    def debug(self, msg):
        logging.log(logging.DEBUG, msg)

