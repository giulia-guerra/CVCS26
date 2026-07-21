# Beyond the Pixel Limit

Computer Vision and Cognitive Systems Project

## Repository structure

- configs/
- data/
- src/
- notebooks/
- scripts_slurm/


## Dataset Pipeline

Supported datasets:
- LIVE
- TID2013
- PIPAL

Each sample returns:

{
 reference_image,
 distorted_image,
 mos,
 image_name
}

Tests:

pytest tests/