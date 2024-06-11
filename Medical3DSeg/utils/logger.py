#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :logger.py
@Author :CodeCat
@Date   :2024/6/11 16:26
"""
import sys
import time

# 定义日志等级
levels = {0: 'ERROR', 1: 'WARNING', 2: 'INFO', 3: 'DEBUG'}
log_level = 2


def log(level=2, message=''):
    current_time = time.time()
    time_array = time.localtime(current_time)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
    if level <= log_level:
        print(
            "{} [{}]\t{}".format(current_time, levels[level], message).encode('utf-8').decode(sys.stdout.encoding)
        )
        sys.stdout.flush()


def debug(message=''):
    log(3, message)


def info(message=''):
    log(2, message)


def waring(message=''):
    log(1, message)


def error(message=''):
    log(0, message)
