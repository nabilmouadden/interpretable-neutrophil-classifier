# Interpretable neutrophil abnormality classifier

An interpretable and calibrated classifier for co-occurring neutrophil morphological
abnormalities in peripheral blood smears.

Given a single neutrophil image from a May–Grünwald–Giemsa stained blood smear, the model

- grades four morphological features between 0 and 1: nuclear segmentation, chromatin
  hypercondensation, granularity and Döhle bodies;
- predicts one or more of seven categories: normal, hypogranulation, hypergranulation,
  hyposegmentation, hypersegmentation, chromatin hypercondensation and Döhle bodies;
- returns calibrated probabilities (per-category temperature scaling) and a conformal
  shortlist of plausible categories, which can be used to send uncertain cells for review.

The encoder is [DinoBloom-B](https://github.com/marrlab/DinoBloom). Each feature has its own
query that attends over the patch tokens, which also gives a per-feature attention map; the
category head sits on the class token. Both heads are trained jointly, with a term that keeps
the predicted categories consistent with the predicted features.

## Installation

Python 3.10 or later. Install [PyTorch](https://pytorch.org) for your platform first, then:

```bash
git clone https://github.com/nabilmouadden/interpretable-neutrophil-classifier.git
cd interpretable-neutrophil-classifier
pip install -e .
```

The DinoBloom-B weights are downloaded from the Hugging Face hub on the first training run.

## Data

Write a CSV manifest with one row per cell and one 0/1 column per category:

```csv
image_path,normal,hypogranulation,hypergranulation,hyposegmentation,hypersegmentation,chromatin_hypercondensation,dohle_bodies
images/cell_0001.png,1,0,0,0,0,0,0
images/cell_0002.png,0,1,0,1,0,0,0
```

Paths are relative to the manifest. An optional `split` column (`train`/`val`/`test`) fixes
the split; otherwise an 80/10/10 multi-label stratified split is drawn per seed. If your
images are sorted into one folder per category, `neutroclf make-manifest --folders DIR --out
manifest.csv` writes the manifest for you.

`external-eval` evaluates a trained model on a single-label dataset such as
[NeuNN](https://doi.org/10.17632/rh3jw43hjs.1); the class mapping is set in
`configs/external_neunn.yaml`.

## Usage

```bash
# train one model (repeat with other seeds)
neutroclf train --config configs/default.yaml --manifest data/manifest.csv --out runs/seed0 --seed 0

# fit temperatures, decision thresholds and the conformal quantile on the validation set
neutroclf calibrate --run runs/seed0

# test-set metrics (F1, AUROC, mAP, ECE, Brier, coverage, referral curve)
neutroclf evaluate --run runs/seed0 --figures

# mean ± SD over several runs
neutroclf summarize runs/seed* --out runs/summary

# predict on new images (JSON with probabilities, shortlist and feature scores)
neutroclf infer --run runs/seed0 path/to/cells/ --out predictions.json

# per-feature attention maps
neutroclf attention --run runs/seed0 --out attention.png cell1.png cell2.png

# external single-label dataset
neutroclf external-eval --run runs/seed0 --manifest data/neunn/manifest.csv --label-map configs/external_neunn.yaml
```

`scripts/train_all_seeds.sh` trains and evaluates five seeds, with and without stain
augmentation. All hyperparameters are in `configs/default.yaml` and can be overridden from the
command line with `--set key=value`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests run on CPU with a small random model and generated toy images.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT
