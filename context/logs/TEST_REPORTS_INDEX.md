# Test Reports Index

This index provides quick access to test reports for the SMPL-X Toolbox project.

## Recent Reports

### 2025-09-08

1. **[UnifiedSmplModel Refactoring Test Report](test-report-20250908-181530-unified-model-refactoring.md)**
   - **Type:** Post-Refactoring Analysis
   - **Component:** core.unified_model
   - **Test Results:** 8/13 passing (61.5%)
   - **Key Issues:** Missing SMPL-H models, beta parameter mismatches
   - **Coverage:** 55% (428/782 lines)
   - **Status:** ⚠️ Configuration issues, core functionality intact

---

## Report Categories

### By Component
- **Core Module**
  - [2025-09-08: unified_model refactoring](test-report-20250908-181530-unified-model-refactoring.md)

### By Test Type
- **Unit Tests**
  - [2025-09-08: test_unified_model.py](test-report-20250908-181530-unified-model-refactoring.md)

### By Status
- **Partial Pass** ⚠️
  - [2025-09-08: unified_model (8/13 passing)](test-report-20250908-181530-unified-model-refactoring.md)

---

## Quick Reference

### Common Issues
1. **Missing Model Files**: SMPL-H models need to be downloaded
2. **Beta Parameters**: Tests assume 10 betas, some models have 16
3. **Legacy Dependencies**: Chumpy requires special installation via pip task

### Test Commands
```bash
# Run all unified model tests
pixi run -e dev python -m pytest unittests/smplx_toolbox/core/test_unified_model.py -v

# Run with coverage
pixi run -e dev python -m pytest unittests/smplx_toolbox/core/test_unified_model.py --cov=smplx_toolbox.core --cov-report=html

# Install legacy dependencies
pixi run -e dev setup-legacy-dev
```

---

*Last Updated: 2025-09-08*