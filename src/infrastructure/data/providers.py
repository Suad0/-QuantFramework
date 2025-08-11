"""
Data providers for multiple market data sources.
"""

import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

from ...domain.exceptions import DataError


@dataclass
class DataProviderConfig:
    """Configuration for data providers."""
    name: str
    enabled: bool = True
    rate_limit: int = 100  # requests per minute
    timeout: int = 30  # seconds
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class IDataProvider(ABC):
    """Interface for market data providers."""
    
    @abstractmethod
    async def fetch_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch market data for given symbols and date range."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and configured."""
        pass
    
    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols (if available)."""
        pass


class YahooFinanceProvider(IDataProvider):
    """Yahoo Finance data provider."""
    
    def __init__(self, config: DataProviderConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._session = None
    
    async def fetch_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        try:
            # Yahoo Finance doesn't support async natively, so we'll use thread pool
            loop = asyncio.get_event_loop()
            
            def _fetch_sync():
                tickers = yf.Tickers(' '.join(symbols))
                data = tickers.history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    prepost=True,
                    threads=True
                )
                return data
            
            data = await loop.run_in_executor(None, _fetch_sync)
            
            if data.empty:
                raise DataError(f"No data returned from Yahoo Finance for symbols: {symbols}")
            
            # Standardize column names and format
            data = self._standardize_data(data, symbols)
            
            self.logger.info(f"Fetched {len(data)} rows from Yahoo Finance for {len(symbols)} symbols")
            return data
            
        except Exception as e:
            self.logger.error(f"Yahoo Finance fetch failed: {str(e)}")
            raise DataError(f"Yahoo Finance data fetch failed: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if Yahoo Finance is available."""
        return self.config.enabled
    
    def get_supported_symbols(self) -> List[str]:
        """Yahoo Finance supports most public equities."""
        return []  # Would need to implement symbol lookup
    
    def _standardize_data(self, data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Standardize data format."""
        if len(symbols) == 1:
            # Single symbol - add symbol column
            data = data.copy()
            data['symbol'] = symbols[0]
        else:
            # Multi-symbol - reshape data
            data = data.stack(level=1).reset_index()
            data.columns = ['date', 'symbol'] + [col for col in data.columns[2:]]
        
        # Ensure consistent column names
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        data = data.rename(columns=column_mapping)
        data['provider'] = 'yahoo'
        data['timestamp'] = pd.Timestamp.now()
        
        return data


class AlphaVantageProvider(IDataProvider):
    """Alpha Vantage data provider."""
    
    def __init__(self, config: DataProviderConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.base_url = config.base_url or "https://www.alphavantage.co/query"
        
        if not config.api_key:
            raise ValueError("Alpha Vantage requires API key")
    
    async def fetch_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch data from Alpha Vantage."""
        all_data = []
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
            for symbol in symbols:
                try:
                    data = await self._fetch_symbol_data(session, symbol, start_date, end_date)
                    if not data.empty:
                        all_data.append(data)
                    
                    # Rate limiting
                    await asyncio.sleep(60 / self.config.rate_limit)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to fetch {symbol} from Alpha Vantage: {str(e)}")
                    continue
        
        if not all_data:
            raise DataError("No data retrieved from Alpha Vantage")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Fetched {len(combined_data)} rows from Alpha Vantage")
        
        return combined_data
    
    async def _fetch_symbol_data(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch data for a single symbol."""
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'outputsize': 'full',
            'apikey': self.config.api_key
        }
        
        async with session.get(self.base_url, params=params) as response:
            if response.status != 200:
                raise DataError(f"Alpha Vantage API error: {response.status}")
            
            data = await response.json()
            
            if 'Error Message' in data:
                raise DataError(f"Alpha Vantage error: {data['Error Message']}")
            
            if 'Note' in data:
                raise DataError(f"Alpha Vantage rate limit: {data['Note']}")
            
            time_series = data.get('Time Series (Daily)', {})
            if not time_series:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            # Standardize columns
            df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividend', 'split']
            df = df.astype(float)
            
            # Add metadata
            df['symbol'] = symbol
            df['provider'] = 'alphavantage'
            df['timestamp'] = pd.Timestamp.now()
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'date'}, inplace=True)
            
            return df
    
    def is_available(self) -> bool:
        """Check if Alpha Vantage is available."""
        return self.config.enabled and bool(self.config.api_key)
    
    def get_supported_symbols(self) -> List[str]:
        """Alpha Vantage supports most US equities."""
        return []


class IEXCloudProvider(IDataProvider):
    """IEX Cloud data provider."""
    
    def __init__(self, config: DataProviderConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.base_url = config.base_url or "https://cloud.iexapis.com/stable"
        
        if not config.api_key:
            raise ValueError("IEX Cloud requires API key")
    
    async def fetch_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch data from IEX Cloud."""
        all_data = []
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
            # IEX supports batch requests
            batch_size = 100  # IEX limit
            
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                try:
                    data = await self._fetch_batch_data(session, batch_symbols, start_date, end_date)
                    if not data.empty:
                        all_data.append(data)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to fetch batch from IEX Cloud: {str(e)}")
                    continue
        
        if not all_data:
            raise DataError("No data retrieved from IEX Cloud")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Fetched {len(combined_data)} rows from IEX Cloud")
        
        return combined_data
    
    async def _fetch_batch_data(
        self,
        session: aiohttp.ClientSession,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch data for multiple symbols in batch."""
        symbols_str = ','.join(symbols)
        
        # Calculate date range for IEX
        days_diff = (end_date - start_date).days
        if days_diff <= 30:
            range_param = '1m'
        elif days_diff <= 90:
            range_param = '3m'
        elif days_diff <= 180:
            range_param = '6m'
        elif days_diff <= 365:
            range_param = '1y'
        else:
            range_param = '5y'
        
        url = f"{self.base_url}/stock/market/batch"
        params = {
            'symbols': symbols_str,
            'types': 'chart',
            'range': range_param,
            'token': self.config.api_key
        }
        
        async with session.get(url, params=params) as response:
            if response.status != 200:
                raise DataError(f"IEX Cloud API error: {response.status}")
            
            data = await response.json()
            
            all_symbol_data = []
            
            for symbol, symbol_data in data.items():
                if 'chart' not in symbol_data:
                    continue
                
                chart_data = symbol_data['chart']
                if not chart_data:
                    continue
                
                df = pd.DataFrame(chart_data)
                df['date'] = pd.to_datetime(df['date'])
                
                # Filter by exact date range
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                
                if df.empty:
                    continue
                
                # Standardize columns
                df = df.rename(columns={
                    'uOpen': 'open',
                    'uHigh': 'high',
                    'uLow': 'low',
                    'uClose': 'close',
                    'uVolume': 'volume'
                })
                
                # Add metadata
                df['symbol'] = symbol
                df['provider'] = 'iex'
                df['timestamp'] = pd.Timestamp.now()
                df['adj_close'] = df['close']  # IEX provides adjusted data by default
                
                all_symbol_data.append(df)
            
            if not all_symbol_data:
                return pd.DataFrame()
            
            return pd.concat(all_symbol_data, ignore_index=True)
    
    def is_available(self) -> bool:
        """Check if IEX Cloud is available."""
        return self.config.enabled and bool(self.config.api_key)
    
    def get_supported_symbols(self) -> List[str]:
        """IEX Cloud supports US equities."""
        return []


class QuandlProvider(IDataProvider):
    """Quandl data provider."""
    
    def __init__(self, config: DataProviderConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.base_url = config.base_url or "https://www.quandl.com/api/v3"
        
        if not config.api_key:
            self.logger.warning("Quandl API key not provided - using free tier limits")
    
    async def fetch_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch data from Quandl."""
        all_data = []
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
            for symbol in symbols:
                try:
                    data = await self._fetch_symbol_data(session, symbol, start_date, end_date)
                    if not data.empty:
                        all_data.append(data)
                    
                    # Rate limiting for free tier
                    if not self.config.api_key:
                        await asyncio.sleep(1)  # Free tier: 50 calls per day
                    
                except Exception as e:
                    self.logger.warning(f"Failed to fetch {symbol} from Quandl: {str(e)}")
                    continue
        
        if not all_data:
            raise DataError("No data retrieved from Quandl")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Fetched {len(combined_data)} rows from Quandl")
        
        return combined_data
    
    async def _fetch_symbol_data(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch data for a single symbol from Quandl."""
        # Quandl uses different database codes - assuming WIKI for US stocks
        database_code = "WIKI"
        url = f"{self.base_url}/datasets/{database_code}/{symbol}.json"
        
        params = {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'order': 'asc'
        }
        
        if self.config.api_key:
            params['api_key'] = self.config.api_key
        
        async with session.get(url, params=params) as response:
            if response.status != 200:
                if response.status == 404:
                    self.logger.warning(f"Symbol {symbol} not found in Quandl")
                    return pd.DataFrame()
                raise DataError(f"Quandl API error: {response.status}")
            
            data = await response.json()
            
            dataset = data.get('dataset', {})
            if not dataset:
                return pd.DataFrame()
            
            column_names = dataset.get('column_names', [])
            data_rows = dataset.get('data', [])
            
            if not data_rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(data_rows, columns=column_names)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Standardize column names (Quandl WIKI format)
            column_mapping = {
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj. Open': 'adj_open',
                'Adj. High': 'adj_high',
                'Adj. Low': 'adj_low',
                'Adj. Close': 'adj_close',
                'Adj. Volume': 'adj_volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Add metadata
            df['symbol'] = symbol
            df['provider'] = 'quandl'
            df['timestamp'] = pd.Timestamp.now()
            
            return df
    
    def is_available(self) -> bool:
        """Check if Quandl is available."""
        return self.config.enabled
    
    def get_supported_symbols(self) -> List[str]:
        """Quandl supports various datasets."""
        return []


class MultiSourceDataProvider:
    """Aggregates multiple data providers with failover logic."""
    
    def __init__(self, providers: List[IDataProvider], failover_enabled: bool = True):
        self.providers = providers
        self.failover_enabled = failover_enabled
        self.logger = logging.getLogger(__name__)
    
    async def fetch_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        preferred_provider: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch data with automatic failover between providers."""
        
        # Filter available providers
        available_providers = [p for p in self.providers if p.is_available()]
        
        if not available_providers:
            raise DataError("No data providers are available")
        
        # Try preferred provider first if specified
        if preferred_provider:
            for provider in available_providers:
                if provider.__class__.__name__.lower().startswith(preferred_provider.lower()):
                    available_providers.remove(provider)
                    available_providers.insert(0, provider)
                    break
        
        last_error = None
        
        for provider in available_providers:
            try:
                self.logger.info(f"Attempting to fetch data using {provider.__class__.__name__}")
                
                data = await provider.fetch_data(symbols, start_date, end_date, **kwargs)
                
                if not data.empty:
                    self.logger.info(f"Successfully fetched data using {provider.__class__.__name__}")
                    return data
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Provider {provider.__class__.__name__} failed: {str(e)}")
                
                if not self.failover_enabled:
                    raise e
                
                continue
        
        # If we get here, all providers failed
        raise DataError(f"All data providers failed. Last error: {str(last_error)}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return [p.__class__.__name__ for p in self.providers if p.is_available()]
    
    def add_provider(self, provider: IDataProvider) -> None:
        """Add a new data provider."""
        self.providers.append(provider)
    
    def remove_provider(self, provider_name: str) -> None:
        """Remove a data provider by name."""
        self.providers = [
            p for p in self.providers 
            if not p.__class__.__name__.lower().startswith(provider_name.lower())
        ]