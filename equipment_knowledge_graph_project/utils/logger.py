#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志工具模块
为整个项目提供统一的日志记录功能
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from loguru import logger

def setup_logger(name: str = None, log_file: str = None, level: str = "INFO") -> logger:
    """设置日志记录器"""
    
    # 创建日志目录
    if log_file is None:
        log_dir = Path("../logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"{name or 'app'}_{timestamp}.log"
    
    # 移除默认的日志处理器
    logger.remove()
    
    # 添加控制台处理器
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # 添加文件处理器
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="1 day",
        retention="30 days",
        compression="zip",
        encoding="utf-8"
    )
    
    return logger

def get_logger(name: str = None) -> logger:
    """获取日志记录器"""
    return logger.bind(name=name)

class LoggerMixin:
    """日志混入类，为其他类提供日志功能"""
    
    @property
    def logger(self):
        """获取日志记录器"""
        return get_logger(self.__class__.__name__)
    
    def log_info(self, message: str, **kwargs):
        """记录信息日志"""
        self.logger.info(message, **kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """记录警告日志"""
        self.logger.warning(message, **kwargs)
    
    def log_error(self, message: str, **kwargs):
        """记录错误日志"""
        self.logger.error(message, **kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """记录调试日志"""
        self.logger.debug(message, **kwargs)
    
    def log_exception(self, message: str, **kwargs):
        """记录异常日志"""
        self.logger.exception(message, **kwargs)