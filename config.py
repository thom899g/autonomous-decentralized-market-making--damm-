"""
Configuration management for DAMM system.
Centralized config with validation and type safety.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv

import logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class ExchangeConfig:
    """Exchange-specific configuration"""
    name: str
    api_key: str = ""
    api_secret: str = ""
    enabled: bool = True
    fee_rate: float = 0.001  # Default 0.1% fee
    rate_limit: int = 10  # Requests per second
    supported_pairs: List[str] = field(default_factory=list)
    
    def validate(self) -> bool:
        """Validate exchange configuration"""
        if not self.name:
            logger.error("Exchange name is required")
            return False
        
        if self.enabled and (not self.api_key or not self.api_secret):
            logger.warning(f"Exchange {self.name} enabled but missing API credentials")
            return False
            
        if self.fee_rate < 0 or self.fee_rate > 0.1:
            logger.error(f"Invalid fee rate for {self.name}: {self.fee_rate}")
            return False
            
        return True

@dataclass
class RLConfig:
    """Reinforcement Learning configuration"""
    model_path: str = "./models/damm_rl_model"
    training_interval: int = 3600  # seconds
    batch_size: int = 32
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    buffer_size: int = 100000
    
    def validate(self) -> bool:
        """Validate RL configuration"""
        if self.learning_rate <= 0:
            logger.error("Learning rate must be positive")
            return False
        if not 0 <= self.gamma <= 1:
            logger.error("Gamma must be between 0 and 1")
            return False
        return True

@dataclass 
class RiskConfig:
    """Risk management configuration"""
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_loss: float = 0.05   # 5% max daily drawdown
    min_spread: float = 0.0005     # 0.05%
    max_slippage: float = 0.001    # 0.1%
    stop_loss_pct: float = 0.02    # 2% stop loss
    take_profit_pct: float = 0.03  # 3% take profit
    max_concurrent_trades: int = 5
    
    def validate(self) -> bool:
        """Validate risk parameters"""
        if self.max_position_size <= 0 or self.max_position_size > 0.5:
            logger.error("Max position size must be between 0 and 0.5")
            return False
        if self.max_daily_loss <= 0 or self.max_daily_loss > 0.2:
            logger.error("Max daily loss must be between 0 and 0.2")
            return False
        return True

class DAMMConfig:
    """Main configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.exchanges: Dict[str, ExchangeConfig] = {}
        self.rl_config = RLConfig()
        self.risk_config = RiskConfig()
        self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[str]) -> None:
        """Load configuration from file or environment"""
        
        # Load exchanges from environment
        self._load_exchanges()
        
        # Load from config file if provided
        if config_path and Path(config