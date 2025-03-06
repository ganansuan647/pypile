#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行所有测试的主脚本
"""

import os
import sys
import unittest
from pathlib import Path

def run_tests() -> None:
    """
    运行tests目录下的所有测试
    
    Returns:
        None
    """
    # 添加项目根目录到Python路径
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # 发现并运行所有测试
    tests_dir = Path(__file__).parent
    test_suite = unittest.defaultTestLoader.discover(tests_dir)
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # 根据测试结果设置退出代码
    sys.exit(not result.wasSuccessful())

if __name__ == "__main__":
    run_tests()
