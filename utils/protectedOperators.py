import numpy as np

# ========================
# Configurable Parameters
# ========================
EPS = 1e-12
CLIP_RANGE = (-1e3, 1e3)
CLIP_RANGE_VALUE = 1e3
PENALTY = 1e6
USE_PENALTY = True  # Toggle: True = hard mode, False = soft mode

def _apply_clip(x, clip_range=CLIP_RANGE):
    return np.clip(x, clip_range[0], clip_range[1])

def _safe_return(valid_mask, value_if_valid, value_if_invalid):
    if USE_PENALTY:
        return np.where(valid_mask, value_if_valid, PENALTY)
    else:
        return np.where(valid_mask, value_if_valid, value_if_invalid)

# ========================
# Protected Operations
# ========================

def protected_divide(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    valid_mask = np.abs(b) > EPS
    result = np.divide(a, b, out=np.zeros_like(a, dtype=float), where=valid_mask)
    return _apply_clip(_safe_return(valid_mask, result, 0.0))

def protected_log(a):
    a = np.asarray(a)
    valid_mask = a > EPS
    safe_input = np.maximum(a, EPS)
    result = np.log(safe_input)
    return _apply_clip(_safe_return(valid_mask, result, np.log(EPS)))

def protected_log2(a):
    a = np.asarray(a)
    valid_mask = a > EPS
    safe_input = np.maximum(a, EPS)
    result = np.log2(safe_input)
    return _apply_clip(_safe_return(valid_mask, result, np.log2(EPS)))

def protected_log10(a):
    a = np.asarray(a)
    valid_mask = a > EPS
    safe_input = np.maximum(a, EPS)
    result = np.log10(safe_input)
    return _apply_clip(_safe_return(valid_mask, result, np.log10(EPS)))

def protected_sqrt(a):
    a = np.asarray(a)
    valid_mask = a >= 0
    safe_input = np.where(valid_mask, a, 0.0)
    result = np.sqrt(safe_input)
    return _apply_clip(_safe_return(valid_mask, result, 0.0))

def protected_power(a, b):
    a = np.asarray(a); b = np.asarray(b)
    with np.errstate(invalid='ignore', over='ignore'):
        result = np.power(a, b)
    valid_mask = np.isfinite(result)
    return _apply_clip(_safe_return(valid_mask, result, 0.0))

def protected_mod(a, b):
    a = np.asarray(a); b = np.asarray(b)
    valid_mask = np.abs(b) > EPS
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(valid_mask, np.mod(a, b), 0.0)
    return _apply_clip(_safe_return(valid_mask, result, 0.0))

def protected_exp(a):
    a = np.asarray(a)
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.exp(a)
    valid_mask = np.isfinite(result)
    return _apply_clip(_safe_return(valid_mask, result, CLIP_RANGE_VALUE))

def protected_tan(a):
    a = np.asarray(a)
    with np.errstate(invalid='ignore'):
        result = np.tan(a)
    valid_mask = np.isfinite(result) & (np.abs(result) <= CLIP_RANGE_VALUE)
    return _apply_clip(_safe_return(valid_mask, result, 0.0))

def protected_reciprocal(a):
    a = np.asarray(a)
    valid_mask = np.abs(a) > EPS
    result = np.where(valid_mask, 1.0 / a, 0.0)
    return _apply_clip(_safe_return(valid_mask, result, 0.0))

def protected_sin(a):
    a = np.asarray(a)
    with np.errstate(invalid='ignore'):
        result = np.sin(a)
    valid_mask = np.isfinite(result)
    return _apply_clip(_safe_return(valid_mask, result, 0.0))

def protected_cos(a):
    a = np.asarray(a)
    with np.errstate(invalid='ignore'):
        result = np.cos(a)
    valid_mask = np.isfinite(result)
    return _apply_clip(_safe_return(valid_mask, result, 0.0))

def protected_arcsin(a):
    a = np.asarray(a)
    valid_mask = (a >= -1) & (a <= 1)
    safe_input = np.clip(a, -1, 1)
    result = np.arcsin(safe_input)
    return _apply_clip(_safe_return(valid_mask, result, 0.0))

def protected_arccos(a):
    a = np.asarray(a)
    valid_mask = (a >= -1) & (a <= 1)
    safe_input = np.clip(a, -1, 1)
    result = np.arccos(safe_input)
    return _apply_clip(_safe_return(valid_mask, result, 0.0))

def protected_arctan(a):
    a = np.asarray(a)
    result = np.arctan(a)
    valid_mask = np.isfinite(result)
    return _apply_clip(_safe_return(valid_mask, result, 0.0))

def protected_sinh(a):
    a = np.asarray(a)
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.sinh(a)
    valid_mask = np.isfinite(result)
    return _apply_clip(_safe_return(valid_mask, result, 0.0))

def protected_cosh(a):
    a = np.asarray(a)
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.cosh(a)
    valid_mask = np.isfinite(result)
    return _apply_clip(_safe_return(valid_mask, result, 0.0))

def protected_tanh(a):
    a = np.asarray(a)
    result = np.tanh(a)
    valid_mask = np.isfinite(result)
    return _apply_clip(_safe_return(valid_mask, result, 0.0))

# ========================
# Safe Op Lists
# ========================
OPERATORS_UNARY = {
    'neg' : np.negative,
    'abs' : np.abs,
    'sqrt' : protected_sqrt,
    'exp' : protected_exp,
    'log' : protected_log,
    'log2' : protected_log2,
    'log10' : protected_log10,
    'sin' : protected_sin,
    'cos' : protected_cos,
    'tan' : protected_tan,
    'asin' : protected_arcsin,
    'acos' : protected_arccos,
    'atan' : protected_arctan,
    'sinh' : protected_sinh,
    'cosh' : protected_cosh,
    'tanh' : protected_tanh,
    'sqr' : np.square,
    'cbrt' : np.cbrt,
    'rec' : protected_reciprocal,
}

OPERATORS_BINARY = {
    'add' : np.add,
    'sub' : np.subtract,
    'mul' : np.multiply,
    'div' : protected_divide,
    'pow' : protected_power,
    'max' : np.maximum,
    'min' : np.minimum,
    'mod' : protected_mod
}

"""
Collection to contain both unary and binary operators.
"""
OPERATORS = { **OPERATORS_UNARY, **OPERATORS_BINARY }
