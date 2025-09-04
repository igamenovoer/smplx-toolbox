# SMPL Model Files

This directory contains SMPL (Skinned Multi-Person Linear) model files.

## Model Files

Place your SMPL model files in this directory:
- `SMPL_NEUTRAL.pkl` - Gender-neutral SMPL model
- `SMPL_MALE.pkl` - Male SMPL model  
- `SMPL_FEMALE.pkl` - Female SMPL model

## Download Instructions

1. Register at https://smpl.is.tue.mpg.de/
2. Download the SMPL model files (version 1.0.0 or 1.1.0)
3. Extract and place the `.pkl` files in this directory

## File Structure

```
smpl/
├── SMPL_NEUTRAL.pkl
├── SMPL_MALE.pkl
└── SMPL_FEMALE.pkl
```

## License

SMPL model files are subject to the SMPL license. Please ensure you have accepted the license terms before using these models.

## Notes

- Model files (`.pkl`, `.npy`, `.npz`) are not tracked by git
- Ensure you have the appropriate permissions to use these models
- The SMPL model contains 10 shape parameters and 23 joints + 1 root joint