# Tests Directory

This directory contains **unit and integration tests** for the trading bot pipeline.

## Test Files

### `test_phase1_improvements.py`
Integration tests for Phase 1 improvements:
- Profit-aware threshold optimization
- Adaptive triple-barrier labeling
- Rolling window training configuration
- Look-ahead bias verification

**Run:**
```bash
cd price_action
python tests/test_phase1_improvements.py
```

### `test_feature_cache.py`
Unit tests for feature caching system:
- Cache hit/miss logic
- TTL expiration
- Cache invalidation
- Memory management

**Run:**
```bash
cd price_action
python tests/test_feature_cache.py
```

### `test_feature_levels.py`
Tests for feature tier system:
- Feature categorization (TIER 0-4)
- Feature loading by tier
- Performance benchmarks

**Run:**
```bash
cd price_action
python tests/test_feature_levels.py
```

## Running All Tests

```bash
cd price_action

# Run all tests
python -m pytest tests/

# Or run individually
python tests/test_phase1_improvements.py
python tests/test_feature_cache.py
python tests/test_feature_levels.py
```

## Adding New Tests

When adding new functionality to the core pipeline, add corresponding tests here:

1. Create `test_<module_name>.py`
2. Use `unittest` or `pytest` framework
3. Test both success and failure cases
4. Include edge cases and boundary conditions
5. Document what is being tested

Example structure:
```python
import unittest

class TestMyFeature(unittest.TestCase):
    def setUp(self):
        # Setup test data
        pass

    def test_normal_case(self):
        # Test expected behavior
        pass

    def test_edge_case(self):
        # Test edge cases
        pass

    def test_error_handling(self):
        # Test error conditions
        pass

if __name__ == '__main__':
    unittest.main()
```

## Test Coverage

Current test coverage focuses on:
- ✅ Phase 1 improvements (profit-aware optimization, adaptive labels)
- ✅ Feature caching system
- ✅ Feature tier management

**TODO:** Add tests for:
- [ ] Model training pipeline
- [ ] Backtester logic
- [ ] Bot trade execution
- [ ] Feature engineering calculations
- [ ] Data preparation pipeline
