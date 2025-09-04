# SMPL-X Model Files

This directory contains SMPL-X (SMPL eXpressive) model files.

## Model Files

Place your SMPL-X model files in this directory:
- `SMPLX_NEUTRAL.pkl` - Gender-neutral SMPL-X model
- `SMPLX_MALE.pkl` - Male SMPL-X model  
- `SMPLX_FEMALE.pkl` - Female SMPL-X model
- `SMPLX_NEUTRAL.npz` - Alternative NPZ format (if available)
- `SMPLX_MALE.npz` - Alternative NPZ format (if available)
- `SMPLX_FEMALE.npz` - Alternative NPZ format (if available)

## Download Instructions

1. Register at https://smpl-x.is.tue.mpg.de/
2. Download the SMPL-X model files (latest version)
3. Extract and place the `.pkl` or `.npz` files in this directory

## File Structure

```
smplx/
├── SMPLX_NEUTRAL.pkl
├── SMPLX_MALE.pkl
├── SMPLX_FEMALE.pkl
└── models/           # Optional: versioned models
    ├── smplx_v1.0/
    └── smplx_v1.1/
```

## Model Details

SMPL-X is the most expressive model, combining:
- **Body**: 21 body joints + 1 pelvis (SMPL base)
- **Hands**: 15 joints per hand (30 total, from MANO)
- **Face**: 51+ facial landmarks (from FLAME)
- **Eyes**: Separate eye pose parameters (3 per eye)

### Parameters
- **Shape**: 10 parameters (body shape)
- **Pose**: 
  - Body: 21 joints × 3 = 63 parameters
  - Hands: 2 × 15 × 3 = 90 parameters
  - Face: 10 expression parameters
  - Jaw: 3 parameters
  - Eyes: 2 × 3 = 6 parameters
- **Translation**: 3 parameters

Total: ~300 parameters for full expressiveness

## License

SMPL-X model files are subject to the Max Planck license. Please ensure you have:
1. Accepted the license terms at https://smpl-x.is.tue.mpg.de/
2. Used the models only for non-commercial scientific research

## Notes

- Model files (`.pkl`, `.npy`, `.npz`) are not tracked by git
- SMPL-X provides the most comprehensive human body representation
- Includes body, hands, face, and eye articulation
- Backwards compatible with SMPL pose parameters