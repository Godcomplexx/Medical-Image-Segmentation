# Two-stage lesion pipeline

This folder is isolated from the current production inference code.

The cascade has two independently trained stages:

1. `binary_detector`: one YOLO segmentation class, `lesion`. Empty YOLO
   labels represent background.
2. `candidate_classifier`: `I`, `P`, `S`, `U`, and `background`. It receives
   an expanded square crop around each detector proposal.

The existing study-level `finetune`/`val` split is preserved. The preparation
scripts do not modify the source dataset.

## 1. Prepare isolated datasets

Run from the repository root in the training virtual environment:

```powershell
python -m finetune.two_stage_pipeline.data
```

Generated data is written to:

```text
finetune/fileStorage/two_stage/
├── detector/
│   ├── dataset.yaml
│   ├── images/train|val/
│   └── labels/train|val/
└── classifier/
    ├── manifest.csv
    └── images/
        ├── train/I|P|S|U|background/
        └── val/I|P|S|U|background/
```

To rebuild generated data explicitly:

```powershell
python -m finetune.two_stage_pipeline.data --overwrite
```

### Full dataset (recommended)

The default build above uses only `train_set/.../finetune` (2 761 slices, of which
318 are `__repeat_` copies). An earlier multiclass-balancing step moved 7 480
legitimate lesion slices into `train_set/excluded_balance`, so the binary detector
was starved of most of its data.

Use `--full` to build the corrected dataset. It merges `excluded_balance` into
train, drops every `__repeat_` copy, leaves validation untouched, and fails loudly
on any train/val basename leakage or duplicate basename across sources:

```powershell
python -m finetune.two_stage_pipeline.data --full --overwrite
```

Output goes to `finetune/fileStorage/two_stage_full/` (the `two_stage` baseline is
left intact). Expected train class counts: `I=477, P=3066, S=7442, U=984` over
9 923 unique slices; validation is unchanged (`I=43, P=700, S=1573, U=179`).
`--keep-repeats` retains the `__repeat_` copies if ever needed.

## 2. Train the binary detector

Baseline recipe on the original dataset:

```powershell
python -m finetune.two_stage_pipeline.train_detector
```

The default recipe uses `yolo11m-seg.pt`, 640 px, no mosaic, and does not
declare background as a YOLO class.

### Full-data control pilot

After building `two_stage_full`, run a 30-epoch pilot at 640 px to isolate the
effect of restoring the data before changing anything else (resolution, 2.5D,
architecture). Compare against the current baseline (Box recall best `0.379`,
Box mAP50 best `0.327`):

```powershell
python -m finetune.two_stage_pipeline.train_detector `
  --data "finetune/fileStorage/two_stage_full/detector/dataset.yaml" `
  --name binary_detector_full_pilot640 `
  --epochs 30
```

If recall is still short of the target, raise small-lesion visibility one lever
at a time — `--imgsz 1024 --batch 2`, then a P2 (stride-4) head or tiling, then a
2.5D `prev/current/next` slice input — re-measuring after each change. Roughly a
third of lesions are smaller than the finest YOLO stride at 640 px, so this is the
main recall bottleneck.

## 3. Mine real detector candidates

Replace the path with the detector's `best.pt`:

```powershell
$detector = "finetune/fileStorage/two_stage/runs/binary_detector/weights/best.pt"

python -m finetune.two_stage_pipeline.mine_candidates --detector $detector --split train
python -m finetune.two_stage_pipeline.mine_candidates --detector $detector --split val
```

Matched predictions are added to their I/P/S/U directories. Unmatched
predictions become hard `background` examples. This reduces the train/inference
crop distribution gap.

Do not mine validation candidates into the training directory.

## 4. Train the classifier

```powershell
python -m finetune.two_stage_pipeline.train_classifier
```

The default classifier is `yolo11m-cls.pt` at 320 px. Ultralytics downloads
the pretrained checkpoint if it is not present locally.

## 4b. Anatomy-aware classifier (recommended stage two)

The I/P/S/U classes are defined by anatomical position, not local appearance
(P abuts a ventricle, U touches the cortex, I is infratentorial, S is neither), so
a tight crop cannot separate them. `anatomy.py` derives geometric context from the
full slice plus slice z-level, and `train_classifier_anatomy.py` fits and evaluates
a classifier on ground-truth lesions — isolating classification quality from
detection:

```powershell
python -m finetune.two_stage_pipeline.train_classifier_anatomy `
  --manifest "finetune/fileStorage/two_stage_full/classifier/manifest.csv"
```

It writes `anatomy_classifier_metrics.json` (per-class precision/recall/F1 +
confusion matrix, learned model vs clinical-rule baseline) and
`anatomy_classifier.joblib`.

First result with FLAIR-proxy features only (no real anatomy masks): `I` F1 0.87
(recall 0.98) confirms z-level captures infratentorial; `P`/`S`/`U` are weaker
because the CSF/cortex proxies are crude. The feature contract in `anatomy.py`
accepts exact FastSurfer/SynthSeg ventricle and cortex masks unchanged — swapping
the proxy for real structure masks is the path to ≥80% on P/S/U.

## 5. Run the cascade

```powershell
$classifier = "finetune/fileStorage/two_stage/runs/candidate_classifier/weights/best.pt"

python -m finetune.two_stage_pipeline.predict `
  --detector $detector `
  --classifier $classifier `
  --source "path/to/image_or_directory" `
  --output "finetune/fileStorage/two_stage/predictions.json" `
  --include-rejected
```

The detector confidence is intentionally low (`0.05`) to protect recall.
The second stage rejects background and candidates below classifier confidence
`0.5`. Tune both thresholds only on the unchanged validation patients.

## 6. Evaluate the complete cascade

```powershell
python -m finetune.two_stage_pipeline.evaluate `
  --detector $detector `
  --classifier $classifier
```

This writes per-class lesion-level TP/FP/FN, precision, recall, and F1 to
`finetune/fileStorage/two_stage/cascade_metrics.json`. A prediction is correct
only when its class is correct and its box reaches IoU 0.5 with an unmatched
ground-truth lesion.

## Interpretation

The classifier cannot recover a lesion missed by the detector. Therefore,
first evaluate binary lesion recall. If I/P/S/U are defined by anatomical
location, the expanded crop may still be insufficient; coordinates, adjacent
slices, or FastSurfer regions should then be added as classifier features.
