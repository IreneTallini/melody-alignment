add # Melody Alignment Project
This project computes chroma features for two audio signals and determines their similarity using DTW.

## Usage
1. `pip install -r requirements.txt`
2. The code loads midi and corresponding real wav data, and applies the following transform:
    1. tempo shift (midi and wav): arg STRETCH_COEFF
    2. white noise addition (midi and wav): arg ADDITIVE_NOISE_STD
    3. random replacement of notes (only wav): arg NOTE_REPLACEMENT_COEFF
2. Run `python melody_metric.py` 
3. The code will print the distance between original and distorted song, toghether with chromagrams and chromagram dynamic time warping optimal path.
