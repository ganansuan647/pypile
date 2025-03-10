#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行所有测试的主脚本（使用pytest）
"""

import sys
import pytest
from pathlib import Path


def run_tests() -> None:
    """
    运行tests目录下的所有测试（使用pytest）

    Returns:
        None
    """
    # 添加项目根目录到Python路径
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # 使用pytest运行所有测试
    tests_dir = Path(__file__).parent
    exit_code = pytest.main([str(tests_dir), "-v"])

    # 根据测试结果设置退出代码
    sys.exit(exit_code)


if __name__ == "__main__":
    run_tests()
