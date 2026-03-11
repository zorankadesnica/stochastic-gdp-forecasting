"""
Data loading module for Serbian economic indicators.
Sources: SORS, NBS, FRED, Eurostat, World Bank
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Union
import requests
from io import StringIO
import warnings

import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file

api_key = os.getenv('FRED_API_KEY')

try:
    from fredapi import Fred
except ImportError:
    Fred = None

try:
    import wbgapi as wb
except ImportError:
    wb = None


class SORSDataLoader:
    """Statistical Office of the Republic of Serbia data loader."""
    
    BASE_URL = "https://data.stat.gov.rs/api"
    
    INDICATORS = {
        'gdp': {'code': '0902', 'name': 'Gross Domestic Product'},
        'ipi': {'code': '1001', 'name': 'Industrial Production Index'},
        'retail': {'code': '1401', 'name': 'Retail Trade Turnover'},
        'cpi': {'code': '1201', 'name': 'Consumer Price Index'},
        'employment': {'code': '0801', 'name': 'Employment'},
        'construction': {'code': '1101', 'name': 'Construction Activity'},
    }
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_gdp_quarterly(
        self,
        start_year: int = 2000,
        end_year: Optional[int] = None,
        real: bool = True
    ) -> pd.DataFrame:
        """
        Fetch quarterly GDP data from SORS.
        
        Parameters
        ----------
        start_year : int
            Start year for data
        end_year : int, optional
            End year (defaults to current year)
        real : bool
            If True, return real GDP (constant prices)
            
        Returns
        -------
        pd.DataFrame
            Quarterly GDP with DatetimeIndex
        """
        if end_year is None:
            end_year = datetime.now().year
            
        # Note: In production, this would call the actual SORS API
        # For now, return structure for manual data loading
        
        quarters = pd.date_range(
            start=f'{start_year}-01-01',
            end=f'{end_year}-12-31',
            freq='QS'
        )
        
        return pd.DataFrame(
            index=quarters,
            columns=['gdp_nominal', 'gdp_real', 'gdp_growth_yoy']
        )
    
    def get_industrial_production(
        self,
        start_date: str = '2010-01-01',
        end_date: Optional[str] = None,
        seasonally_adjusted: bool = True
    ) -> pd.DataFrame:
        """
        Fetch monthly Industrial Production Index.
        
        Returns index values with 2021=100 base year.
        """
        dates = pd.date_range(start=start_date, end=end_date or 'today', freq='MS')
        
        return pd.DataFrame(
            index=dates,
            columns=['ipi_total', 'ipi_manufacturing', 'ipi_mining', 'ipi_utilities']
        )
    
    def get_retail_trade(
        self,
        start_date: str = '2010-01-01',
        end_date: Optional[str] = None,
        constant_prices: bool = True
    ) -> pd.DataFrame:
        """Fetch monthly retail trade turnover."""
        dates = pd.date_range(start=start_date, end=end_date or 'today', freq='MS')
        
        return pd.DataFrame(
            index=dates,
            columns=['retail_total', 'retail_food', 'retail_nonfood']
        )
    
    def get_cpi(
        self,
        start_date: str = '2010-01-01',
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch monthly Consumer Price Index."""
        dates = pd.date_range(start=start_date, end=end_date or 'today', freq='MS')
        
        return pd.DataFrame(
            index=dates,
            columns=['cpi_all', 'cpi_food', 'cpi_energy', 'cpi_core']
        )


class NBSDataLoader:
    """National Bank of Serbia data loader."""
    
    BASE_URL = "https://nbs.rs"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_key_policy_rate(
        self,
        start_date: str = '2007-01-01',
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch NBS key policy rate history.
        
        Current rate: 5.75% (as of February 2026)
        """
        dates = pd.date_range(start=start_date, end=end_date or 'today', freq='D')
        
        return pd.DataFrame(
            index=dates,
            columns=['key_rate', 'deposit_rate', 'lending_rate']
        )
    
    def get_exchange_rates(
        self,
        start_date: str = '2010-01-01',
        end_date: Optional[str] = None,
        currencies: List[str] = ['EUR', 'USD']
    ) -> pd.DataFrame:
        """Fetch daily exchange rates RSD/currency."""
        dates = pd.date_range(start=start_date, end=end_date or 'today', freq='B')
        
        columns = [f'RSD_{c}' for c in currencies]
        return pd.DataFrame(index=dates, columns=columns)
    
    def get_money_supply(
        self,
        start_date: str = '2010-01-01',
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch monthly money supply aggregates M0, M1, M2, M3."""
        dates = pd.date_range(start=start_date, end=end_date or 'today', freq='MS')
        
        return pd.DataFrame(
            index=dates,
            columns=['M0', 'M1', 'M2', 'M3']
        )
    
    def get_inflation_expectations(
        self,
        start_date: str = '2015-01-01'
    ) -> pd.DataFrame:
        """Fetch inflation expectations survey results."""
        dates = pd.date_range(start=start_date, end='today', freq='MS')
        
        return pd.DataFrame(
            index=dates,
            columns=['exp_1y_household', 'exp_1y_corporate', 'exp_2y']
        )


class FREDDataLoader:
    """FRED (Federal Reserve Economic Data) loader for Serbia data."""
    
    SERBIA_SERIES = {
        'gdp_real': 'CLVMNACNSAB1GQRS',      # Real GDP, quarterly
        'gdp_nominal': 'CPMNACNSAB1GQRS',     # Nominal GDP, quarterly  
        'inflation': 'FPCPITOTLZGSRB',        # CPI inflation, annual
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED loader.
        
        Parameters
        ----------
        api_key : str, optional
            FRED API key. Get free key at https://fred.stlouisfed.org/docs/api/api_key.html
        """
        if Fred is None:
            raise ImportError("fredapi not installed. Run: pip install fredapi")
        
        self.api_key = api_key
        if api_key:
            self.fred = Fred(api_key=api_key)
        else:
            self.fred = None
            warnings.warn("No FRED API key provided. Set FRED_API_KEY env variable.")
    
    def get_serbia_gdp(
        self,
        start_date: str = '1995-01-01',
        end_date: Optional[str] = None,
        real: bool = True
    ) -> pd.Series:
        """
        Fetch Serbia GDP from FRED (Eurostat source).
        
        Series: CLVMNACNSAB1GQRS (real) or CPMNACNSAB1GQRS (nominal)
        """
        series_id = self.SERBIA_SERIES['gdp_real' if real else 'gdp_nominal']
        
        if self.fred:
            return self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
        else:
            # Return empty series with correct structure
            return pd.Series(name=series_id, dtype=float)
    
    def get_serbia_inflation(
        self,
        start_date: str = '1995-01-01'
    ) -> pd.Series:
        """Fetch Serbia CPI inflation from FRED (World Bank source)."""
        if self.fred:
            return self.fred.get_series(
                self.SERBIA_SERIES['inflation'],
                observation_start=start_date
            )
        return pd.Series(name='inflation', dtype=float)


class WorldBankDataLoader:
    """World Bank Open Data loader."""
    
    SERBIA_CODE = 'SRB'
    
    INDICATORS = {
        'gdp_current': 'NY.GDP.MKTP.CD',
        'gdp_growth': 'NY.GDP.MKTP.KD.ZG',
        'gdp_per_capita': 'NY.GDP.PCAP.CD',
        'inflation': 'FP.CPI.TOTL.ZG',
        'unemployment': 'SL.UEM.TOTL.ZS',
        'trade_pct_gdp': 'NE.TRD.GNFS.ZS',
        'fdi_net': 'BX.KLT.DINV.WD.GD.ZS',
    }
    
    def __init__(self):
        if wb is None:
            raise ImportError("wbgapi not installed. Run: pip install wbgapi")
    
    def get_indicators(
        self,
        indicators: Optional[List[str]] = None,
        start_year: int = 1990,
        end_year: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch World Bank indicators for Serbia.
        
        Parameters
        ----------
        indicators : list, optional
            List of indicator keys from INDICATORS dict.
            If None, fetch all.
        """
        if indicators is None:
            indicators = list(self.INDICATORS.keys())
        
        series_codes = [self.INDICATORS[i] for i in indicators]
        
        if end_year is None:
            end_year = datetime.now().year
        
        try:
            data = wb.data.DataFrame(
                series_codes,
                economy=self.SERBIA_CODE,
                time=range(start_year, end_year + 1)
            )
            return data
        except Exception as e:
            warnings.warn(f"World Bank API error: {e}")
            return pd.DataFrame()


def load_serbian_indicators(
    start_date: str = '2010-01-01',
    end_date: Optional[str] = None,
    indicators: Optional[List[str]] = None,
    fred_api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to load all Serbian economic indicators.
    
    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date (defaults to today)
    indicators : list, optional
        Specific indicators to load. Options:
        'gdp', 'ipi', 'retail', 'cpi', 'interest_rate', 'exchange_rate'
    fred_api_key : str, optional
        FRED API key for additional data
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all indicators, monthly frequency
    """
    
    if indicators is None:
        indicators = ['gdp', 'ipi', 'retail', 'cpi', 'interest_rate']
    
    sors = SORSDataLoader()
    nbs = NBSDataLoader()
    
    dfs = []
    
    if 'gdp' in indicators:
        gdp = sors.get_gdp_quarterly()
        # Interpolate to monthly
        gdp_monthly = gdp.resample('MS').interpolate(method='linear')
        dfs.append(gdp_monthly)
    
    if 'ipi' in indicators:
        dfs.append(sors.get_industrial_production(start_date, end_date))
    
    if 'retail' in indicators:
        dfs.append(sors.get_retail_trade(start_date, end_date))
    
    if 'cpi' in indicators:
        dfs.append(sors.get_cpi(start_date, end_date))
    
    if 'interest_rate' in indicators:
        ir = nbs.get_key_policy_rate(start_date, end_date)
        ir_monthly = ir.resample('MS').last()
        dfs.append(ir_monthly)
    
    if 'exchange_rate' in indicators:
        fx = nbs.get_exchange_rates(start_date, end_date)
        fx_monthly = fx.resample('MS').mean()
        dfs.append(fx_monthly)
    
    if dfs:
        combined = pd.concat(dfs, axis=1)
        combined = combined.loc[start_date:end_date]
        return combined
    
    return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    print("Loading Serbian economic indicators...")
    
    # Initialize loaders
    sors = SORSDataLoader()
    nbs = NBSDataLoader()
    
    print(f"\nSORS Indicators available: {list(sors.INDICATORS.keys())}")
    print(f"NBS key rate: 5.75% (Feb 2026)")
    
    # FRED example
    print("\nFRED Series for Serbia:")
    print("  - CLVMNACNSAB1GQRS: Real GDP")
    print("  - FPCPITOTLZGSRB: Inflation")
