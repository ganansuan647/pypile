#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyPile 命令行入口点
"""

from .pile_manager import PileManager

def main() -> None:
    """主程序入口点"""
    pile = PileManager()
    pile.cli()

if __name__ == "__main__":
    main()
