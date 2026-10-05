# Business Entity Resolution

## Reproduce the submitted outputs

Run from the project root, where the `dataset/` directory is available:

```powershell
python -m pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/train_and_predict.py
```

The script trains a LightGBM classifier from `dataset/train/` and performs inference using `dataset/test/`. It writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. It processes country labels dynamically, including test-only countries such as France. The script trains a fresh model each run; no model checkpoint is required.

The files under `dataset/train/` and `dataset/test/` must retain the challenge-provided names and tab-separated format. The GitHub source repository does not include the datasets or generated output files; provide the challenge datasets at the project root before running the command above.

## Which script produced the package outputs?

`src/train_and_predict.py` is the authoritative end-to-end training and inference script used for the included predictions. `src/pipeline.py` is an alternative heuristic inference implementation; it does not train the LightGBM model and is not the script used to create the included predictions.

## Runtime notes

The script loads country-level test targets and retains candidate/scoring results in memory. Plan for substantial available memory when running on the full challenge dataset. The training split is stratified and reproducible (`random_state=42`); LightGBM is configured to use all available CPU workers.
