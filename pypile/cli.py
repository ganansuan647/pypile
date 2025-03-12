#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyPile 命令行入口点
"""

from .pile_manager import PileManager

def pypile_cli() -> None:
    """cli主程序入口点"""
    pile = PileManager(welcome=False)
    pile.cli()

if __name__ == "__main__":
    pypile_cli()
