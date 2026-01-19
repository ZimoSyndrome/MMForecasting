from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class MarketData(BaseModel):
    """
    Canonical representation of a single OHLCV bar.
    """
    timestamp: datetime
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)
    adj_close: Optional[float] = Field(None, gt=0, description="Adjusted close price if available")

    class Config:
        frozen = True  # Immutable to prevent accidental changes

class TickerData(BaseModel):
    """
    Container for a sequence of market data for a specific ticker.
    """
    ticker: str
    data: List[MarketData]
    source: str = Field(..., description="Source of the data (e.g., 'alpaca', 'yahoo')")
    metadata: Optional[dict] = Field(default_factory=dict)
