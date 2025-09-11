# Test plans for keypoint matching feature

This document outlines the test plans for the keypoint matching feature in the SMPL-X toolbox. The goal is to ensure that all components of the keypoint matching functionality are thoroughly tested and validated.

## Manual Testing Cases

### Case 1: Basic 3d Keypoint Matching

we are going to randomly generate some 3d keypoints and fit the smplx model to them.

this is a smoke test, DO NOT use pytest or unittest framework, just write a simple script that can be run in jupyter notebook or python shell.

to run any python code, use `pixi run -e dev`

#### what models to use
- target model: SMPL-X and SMPL-H
- initial pose: identity pose (all zeros)

#### how to generate key point
- select keypoints of left hand and right foot
- target keypoints = neutral_model.keypoints + noise (randn() *0.1)

#### visualization
show the followings:
- original keypoints (black) and mesh (gray=0.5, wireframe)
- target keypoints (red)
- fitted keypoints (green) and posed mesh (blue, wireframe)

#### where to put the code
- for temporary code, put them in `tmp/` folder.
- for smoke test, put them in `tests/smoke_test_*.py` files.