"""
Utility functions for randomization policies.

Key functions:
- Seed generation (hierarchical, deterministic)
- Exposure calculations (net exposure, tolerance checks)
- Market hours handling (checking, clamping)

Owner: P4
Week: 2
"""

from datetime import datetime, time
from typing import Dict, Optional
import hashlib


# ============================================================================
# Seed Generation (Hierarchical & Deterministic)
# ============================================================================

def generate_policy_seed(
    master_seed: int, 
    policy_name: str, 
    date: Optional[datetime] = None,
    symbol: Optional[str] = None
) -> int:
    """
    Generate deterministic seeds with hierarchy.
    
    Hierarchy:
    - Master seed (e.g., 42) → Policy seed
    - Master + date → Daily seed
    - Master + date + symbol → Trade-level seed
    
    Critical Design: Use SHA1 hash for determinism across platforms.
    
    Args:
        master_seed: Top-level seed (e.g., 42)
        policy_name: 'Uniform', 'OU', or 'Pink'
        date: Optional datetime for daily seeds
        symbol: Optional symbol for trade-level seeds
    
    Returns:
        Derived seed in int32 range (0 to 2^31-1)
    
    Examples:
        >>> # Policy-level seed
        >>> generate_policy_seed(42, 'Uniform')
        1234567890
        
        >>> # Same inputs → same output (deterministic)
        >>> s1 = generate_policy_seed(42, 'Uniform')
        >>> s2 = generate_policy_seed(42, 'Uniform')
        >>> assert s1 == s2
        
        >>> # Daily seed
        >>> from datetime import datetime
        >>> generate_policy_seed(42, 'Uniform', date=datetime(2025, 7, 15))
        987654321
        
        >>> # Trade-level seed
        >>> generate_policy_seed(42, 'Uniform', 
        ...                      date=datetime(2025, 7, 15), 
        ...                      symbol='AAPL')
        555555555
    """
    # Build deterministic string from inputs
    seed_components = [str(master_seed), policy_name]
    
    if date is not None:
        seed_components.append(date.strftime('%Y%m%d'))
    
    if symbol is not None:
        seed_components.append(symbol)
    
    seed_string = '|'.join(seed_components)
    
    # Hash to get deterministic integer
    # Critical: Use SHA1 (not Python's hash()) for cross-platform determinism
    hash_obj = hashlib.sha1(seed_string.encode('utf-8'))
    hash_int = int(hash_obj.hexdigest(), 16)
    
    # Clamp to int32 range for numpy compatibility
    return hash_int % (2**31)


# ============================================================================
# Exposure Calculations
# ============================================================================

def calculate_net_exposure(positions: Dict[str, float]) -> float:
    """
    Calculate net exposure from position dictionary.
    
    Convention:
    - Positive values = long positions
    - Negative values = short positions
    - Net exposure = sum of all positions
    
    Args:
        positions: {symbol: shares} dictionary
                   Example: {'AAPL': 100, 'MSFT': -50}
    
    Returns:
        Net exposure (sum of all positions)
    
    Examples:
        >>> # Balanced portfolio
        >>> positions = {'AAPL': 100, 'MSFT': -100}
        >>> calculate_net_exposure(positions)
        0.0
        
        >>> # Net long
        >>> positions = {'AAPL': 100, 'MSFT': -50, 'GOOGL': -30}
        >>> calculate_net_exposure(positions)
        20.0
        
        >>> # Empty portfolio
        >>> calculate_net_exposure({})
        0.0
    """
    if not positions:
        return 0.0
    
    return float(sum(positions.values()))


def calculate_gross_exposure(positions: Dict[str, float]) -> float:
    """
    Calculate gross exposure (sum of absolute values).
    
    Gross exposure measures total capital deployed regardless of direction.
    
    Args:
        positions: {symbol: shares} dictionary
    
    Returns:
        Gross exposure (sum of absolute values)
    
    Examples:
        >>> positions = {'AAPL': 100, 'MSFT': -50, 'GOOGL': -30}
        >>> calculate_gross_exposure(positions)
        180.0
    """
    if not positions:
        return 0.0
    
    return float(sum(abs(v) for v in positions.values()))


def is_within_exposure_tolerance(
    positions_before: Dict[str, float],
    positions_after: Dict[str, float],
    tolerance: float = 0.05
) -> bool:
    """
    Check if net exposure change is within tolerance.
    
    Critical Design Decision: Absolute tolerance (not relative).
    Rationale: Target net exposure is 0, so relative tolerance doesn't make sense.
    
    Args:
        positions_before: Positions before perturbation
        positions_after: Positions after perturbation
        tolerance: Maximum allowed absolute change (default 0.05 = 5 shares)
    
    Returns:
        True if within tolerance, False otherwise
    
    Examples:
        >>> before = {'AAPL': 100, 'MSFT': -100}  # Net = 0
        >>> after = {'AAPL': 102, 'MSFT': -98}    # Net = 4
        >>> is_within_exposure_tolerance(before, after, tolerance=5.0)
        True
        
        >>> # Large violation
        >>> after = {'AAPL': 120, 'MSFT': -80}    # Net = 40
        >>> is_within_exposure_tolerance(before, after, tolerance=5.0)
        False
        
        >>> # Edge case: exactly at tolerance
        >>> after = {'AAPL': 105, 'MSFT': -100}   # Net = 5
        >>> is_within_exposure_tolerance(before, after, tolerance=5.0)
        True  # Uses <= not <
    """
    net_before = calculate_net_exposure(positions_before)
    net_after = calculate_net_exposure(positions_after)
    
    delta = abs(net_after - net_before)
    
    # Use <= to include boundary cases
    return delta <= tolerance


# ============================================================================
# Market Hours Handling
# ============================================================================

def check_market_hours(
    timestamp: datetime,
    market_open: time = time(9, 30),
    market_close: time = time(16, 0)
) -> bool:
    """
    Check if timestamp falls within market hours.
    
    Default US equity market hours: 9:30 AM - 4:00 PM ET
    
    Args:
        timestamp: Datetime to check
        market_open: Market open time (default 9:30 AM)
        market_close: Market close time (default 4:00 PM)
    
    Returns:
        True if within market hours, False otherwise
    
    Examples:
        >>> from datetime import datetime
        >>> # During market hours
        >>> ts = datetime(2025, 7, 15, 10, 30)
        >>> check_market_hours(ts)
        True
        
        >>> # Before market open
        >>> ts = datetime(2025, 7, 15, 8, 0)
        >>> check_market_hours(ts)
        False
        
        >>> # After market close
        >>> ts = datetime(2025, 7, 15, 17, 0)
        >>> check_market_hours(ts)
        False
        
        >>> # Exactly at open (inclusive)
        >>> ts = datetime(2025, 7, 15, 9, 30)
        >>> check_market_hours(ts)
        True
        
        >>> # Exactly at close (inclusive)
        >>> ts = datetime(2025, 7, 15, 16, 0)
        >>> check_market_hours(ts)
        True
    """
    ts_time = timestamp.time()
    return market_open <= ts_time <= market_close


def clamp_to_market_hours(
    timestamp: datetime,
    market_open: time = time(9, 30),
    market_close: time = time(16, 0)
) -> datetime:
    """
    Clamp timestamp to market hours if outside bounds.
    
    Critical Logic:
    - If before open → clamp to open
    - If after close → clamp to close
    - If already within hours → return unchanged
    
    Args:
        timestamp: Datetime to clamp
        market_open: Market open time
        market_close: Market close time
    
    Returns:
        Clamped datetime within market hours
    
    Examples:
        >>> from datetime import datetime
        >>> # Before market open
        >>> ts = datetime(2025, 7, 15, 8, 0)
        >>> clamped = clamp_to_market_hours(ts)
        >>> clamped.time()
        datetime.time(9, 30)
        
        >>> # After market close
        >>> ts = datetime(2025, 7, 15, 17, 30)
        >>> clamped = clamp_to_market_hours(ts)
        >>> clamped.time()
        datetime.time(16, 0)
        
        >>> # Already within hours
        >>> ts = datetime(2025, 7, 15, 12, 0)
        >>> clamped = clamp_to_market_hours(ts)
        >>> clamped == ts
        True
    """
    ts_time = timestamp.time()
    
    # Before market open → clamp to open
    if ts_time < market_open:
        return timestamp.replace(
            hour=market_open.hour,
            minute=market_open.minute,
            second=0,
            microsecond=0
        )
    
    # After market close → clamp to close
    if ts_time > market_close:
        return timestamp.replace(
            hour=market_close.hour,
            minute=market_close.minute,
            second=0,
            microsecond=0
        )
    
    # Already within hours
    return timestamp


# ============================================================================
# Parameter Validation
# ============================================================================

def validate_parameter_bounds(
    param_name: str,
    value: float,
    min_value: float,
    max_value: float,
    warn_only: bool = True
) -> float:
    """
    Validate parameter is within bounds.
    
    Critical Design: Soft bounds with warnings (not hard limits).
    Rationale: Allow P7's adaptive adversary to explore full range.
    
    Args:
        param_name: Name of parameter (for error messages)
        value: Parameter value to check
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        warn_only: If True, log warning and clamp. If False, raise error.
    
    Returns:
        Clamped value within bounds
    
    Raises:
        ValueError: If warn_only=False and value out of bounds
    
    Examples:
        >>> # Within bounds
        >>> validate_parameter_bounds('timing_range', 2.0, 0.1, 6.5)
        2.0
        
        >>> # Below minimum (with warning)
        >>> validate_parameter_bounds('timing_range', 0.05, 0.1, 6.5)
        0.1
        
        >>> # Above maximum (with warning)
        >>> validate_parameter_bounds('timing_range', 10.0, 0.1, 6.5)
        6.5
    """
    if value < min_value or value > max_value:
        msg = f"{param_name}={value} outside bounds [{min_value}, {max_value}]"
        
        if warn_only:
            import warnings
            warnings.warn(f"{msg}. Clamping to bounds.")
            return max(min_value, min(value, max_value))
        else:
            raise ValueError(msg)
    
    return value