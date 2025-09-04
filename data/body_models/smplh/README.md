# SMPL-H Model Files

This directory contains SMPL-H (SMPL with Hands) model files.

## Model Files

Place your SMPL-H model files in this directory:
- `SMPLH_NEUTRAL.pkl` - Gender-neutral SMPL-H model
- `SMPLH_MALE.pkl` - Male SMPL-H model
- `SMPLH_FEMALE.pkl` - Female SMPL-H model

## Download Instructions

1. Register at https://mano.is.tue.mpg.de/
2. Download the extended SMPL+H model files
3. Extract and place the `.pkl` files in this directory

## File Structure

```
smplh/
├── SMPLH_NEUTRAL.pkl
├── SMPLH_MALE.pkl
└── SMPLH_FEMALE.pkl
```

## Model Details

SMPL-H extends the SMPL model with:
- Articulated hand models (MANO)
- 15 joints per hand (30 additional joints total)
- Total of 52 joints (22 body + 30 hand joints)
- Hand pose parameters: 2×15×3 = 90 parameters

## License

SMPL-H model files are subject to the MPI license. Please ensure you have accepted the license terms before using these models.

## Notes

- Model files (`.pkl`, `.npy`, `.npz`) are not tracked by git
- SMPL-H includes both body and detailed hand articulation
- Compatible with MANO hand model parameters