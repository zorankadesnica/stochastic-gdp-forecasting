#!/usr/bin/env python3
"""
Data download script for Serbian economic indicators.

Sources:
- FRED: fred.stlouisfed.org (CLVMNACNSAB1GQRS, FPCPITOTLZGSRB)
- World Bank: data.worldbank.org/country/RS
- SORS: data.stat.gov.rs (manual download)
- NBS: nbs.rs/statistika (manual download)
"""

import os
from pathlib import Path

RAW_DIR = Path(__file__).parent / 'raw'
RAW_DIR.mkdir(exist_ok=True)

print("Data Download Script")
print("=" * 50)

# Instructions
print("""
AUTOMATED (requires API keys):
  FRED_API_KEY: Get at fred.stlouisfed.org/docs/api/api_key.html
  
  $ export FRED_API_KEY=your_key
  $ python -c "from src.data_loader import FREDDataLoader; ..."

MANUAL DOWNLOADS:

1. SORS (data.stat.gov.rs):
   - GDP quarterly: National Accounts section
   - IPI monthly: Industry section  
   - Retail: Trade section
   - CPI: Prices section
   
2. NBS (nbs.rs/statistika):
   - Statistical Bulletin (Excel)
   - Interest rates, Exchange rates
   
3. Eurostat (ec.europa.eu/eurostat):
   - Search "Serbia" for candidate country data

Save all files to: data/raw/
""")
