# Action Items - Test Failures Resolution
**Generated:** 2025-09-08  
**Priority:** High  
**Component:** UnifiedSmplModel Tests

## 🔴 Critical (P0) - Blocks Testing

### 1. Fix Beta Parameter Mismatch
**Issue:** Tests hardcode 10 betas, but SMPL-X model expects 16  
**Files to Modify:**
- `unittests/smplx_toolbox/core/test_unified_model.py` (line 87)

**Solution:**
```python
@pytest.fixture
def batch2_inputs(smplx_model) -> UnifiedSmplInputs:
    # Query actual beta count from model
    uni = UnifiedSmplModel.from_smpl_model(smplx_model)
    num_betas = uni.num_betas
    return UnifiedSmplInputs(
        betas=torch.randn(2, num_betas),  # Dynamic count
        root_orient=torch.randn(2, 3),
        pose_body=torch.randn(2, 63),
    )
```

### 2. Handle Missing SMPL-H Model
**Issue:** SMPL-H model file not present  
**Missing File:** `data/body_models/smplh/SMPLH_NEUTRAL.pkl`

**Options:**
1. Download SMPL-H models from official source
2. Add pytest skip decorator:
```python
@pytest.mark.skipif(
    not Path("data/body_models/smplh/SMPLH_NEUTRAL.pkl").exists(),
    reason="SMPL-H model not available"
)
```

## 🟡 Important (P1) - Improves Stability

### 3. Update Deprecation Warnings
**Files with Issues:**
- `context/refcode/smplx/smplx/body_models.py:142` - scipy.sparse.csc
- `context/refcode/smplx/smplx/utils.py:117` - numpy array conversion
- Chumpy package - LinearOperator import

**Actions:**
- Update scipy imports to use `scipy.sparse` namespace
- Fix numpy 2.0 compatibility issues
- Consider chumpy alternatives or fork

### 4. Improve Test Adaptability
**Current Issues:**
- Tests assume fixed vertex counts
- Tests assume fixed joint counts
- Tests assume fixed beta counts

**Improvements Needed:**
```python
# Query model for actual values
expected_vertices = model.v_template.shape[0]
expected_joints = len(model.joint_names) if hasattr(model, 'joint_names') else 55
expected_betas = model.num_betas
```

## 🟢 Nice to Have (P2) - Long-term Quality

### 5. Increase Test Coverage
**Current:** 55% (428/782 lines)  
**Target:** 80%+

**Critical Gaps:**
- Joint unification logic (lines 385-453)
- Full pose computation (lines 491-526)
- Device movement functionality

### 6. Documentation Updates
**Create/Update:**
- Model setup guide in `docs/setup/models.md`
- Test data requirements in `tests/README.md`
- Troubleshooting guide for common issues

## 🔧 Setup Instructions

### For Developers Starting Fresh:

1. **Install Dependencies:**
```bash
pixi install
pixi run -e dev setup-legacy-dev  # Installs chumpy
```

2. **Download Model Files:**
Place in `data/body_models/`:
- `smpl/SMPL_NEUTRAL.pkl` ✅ (present)
- `smplh/SMPLH_NEUTRAL.pkl` ❌ (needed)
- `smplx/SMPLX_NEUTRAL.npz` ✅ (present)

3. **Run Tests:**
```bash
# Quick test
pixi run -e dev python -m pytest unittests/smplx_toolbox/core/test_unified_model.py -v

# With coverage
pixi run -e dev test-cov
```

## 📊 Success Criteria

Tests should achieve:
- [ ] All 13 tests passing
- [ ] No deprecation warnings from our code
- [ ] Coverage > 80%
- [ ] Clear skip messages for missing models
- [ ] Adaptive to different model configurations

## 📝 Notes

- Chumpy is installed via pip task, not pypi-dependencies (legacy setup.py issue)
- Reference implementation (context/refcode/smplx) has its own deprecation issues
- Model files are not in version control due to licensing

## 🔗 References

- [Detailed Test Report](test-report-20250908-181530-unified-model-refactoring.md)
- [SMPL Model Page](https://smpl.is.tue.mpg.de/)
- [Chumpy GitHub](https://github.com/mattloper/chumpy)

---

*Priority: Focus on P0 items first to unblock testing, then address P1 for stability*