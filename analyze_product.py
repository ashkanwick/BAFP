#!/usr/bin/env python3
"""
Digikala Product Sentiment & Purchase Advisor CLI.
==================================================

Usage:
    python analyze_product.py "https://www.digikala.com/product/dkp-18575915/"
    python analyze_product.py dkp-18575915
    python analyze_product.py --max-comments 50
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.product_advisor import main

if __name__ == "__main__":
    sys.exit(main())
