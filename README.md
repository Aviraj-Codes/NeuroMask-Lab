# NeuroMask Lab - Brain Tumor Segmentation with a 2D U-Net

A reproducible student project for segmenting the **whole tumor** in multimodal brain MRI. It uses a compact PyTorch U-Net on axial slices from the Medical Segmentation Decathlon (MSD) Task01 BrainTumour dataset.

## Scope and dataset choice

- **Task:** binary whole-tumor segmentation (all nonzero annotation labels combined).
- **Inputs:** the four co-registered MRI channels supplied per patient, ordered FLAIR, T1w, T1gd, T2w. The code expects channel-last NIfTI volumes shaped `[X, Y, Z, 4]`.
- **Model:** 2D U-Net, trained slice by slice to keep GPU memory and implementation complexity manageable.
- **Data:** MSD Task01 BrainTumour, a BraTS-derived collection with 484 labeled training cases and 266 unlabeled test cases. The project makes its own seeded 70/15/15 patient-level train/validation/test split from the labeled cases. The official unlabeled test cases are not used.
- **Subset:** the downloader defaults to 20 labeled patients for a practical first run. Increase `--cases` to fetch more, or set `--max-cases 0` during training to use all cases already downloaded. Small-subset results are for learning and pipeline verification, not benchmark results.

The dataset is available from the Medical Segmentation Decathlon download page and its public AWS mirror; the AWS Open Data Registry lists **CC BY-SA 4.0**, no-account access, and both archived and uncompressed data. The smaller downloader fetches just selected labeled cases. Do not redistribute the archive or derived dataset without following the license. Preserve attribution and share-alike terms for dataset-derived material.

**Attribution:** Simpson et al., *A large annotated medical image dataset for the development and evaluation of segmentation algorithms* (2019), and Antonelli et al., *The Medical Segmentation Decathlon* (2022), DOI: [10.1038/s41467-022-30695-9](https://doi.org/10.1038/s41467-022-30695-9). Also retain the dataset's supplied `LICENSE.txt` and cite the BraTS source papers listed in its dataset documentation when publishing work.

## Project structure

```text
brain-tumor-segmentation/
├── data/                         # downloaded data (not committed)
│   └── Task01_BrainTumour/
│       ├── imagesTr/
│       ├── labelsTr/
│       ├── imagesTs/
│       └── dataset.json
├── outputs/                      # local plots, metrics, and checkpoints
├── app.py                         # Streamlit inference app
├── scripts/
│   ├── download_subset.py
│   ├── explore_data.py
│   ├── train.py
│   └── predict.py
├── src/brain_tumor_seg/
│   ├── data.py
│   ├── metrics.py
│   └── model.py
├── requirements.txt
└── README.md
```

## 1. Set up the environment

Python 3.10 or 3.11 is recommended. A CUDA-enabled PyTorch install can speed up training; install the PyTorch build appropriate for your computer first if you need GPU support, then install the remaining requirements.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.

## 2. Obtain the data

For the **small download**, the script inspects the official public tar archive through S3 byte-range requests and fetches only the selected image and mask contents (no AWS account or full archive download required):

```powershell
python scripts/download_subset.py --cases 20 --seed 42
```

This downloads paired image and mask files for selected cases and reports their actual combined size. Change `--cases` and `--seed` to choose a different reproducible subset. It reads the tar index remotely and requests byte ranges for selected files; it does not download the full archive. If S3 range access is unavailable, the fallback is downloading the full `Task01_BrainTumour.tar` archive from the [official MSD data page](https://medicaldecathlon.com/dataaws/), around 7.6 GB:

```powershell
aws s3 cp --no-sign-request s3://msd-for-monai/Task01_BrainTumour.tar data/Task01_BrainTumour.tar
New-Item -ItemType Directory -Force data | Out-Null
tar -xf data/Task01_BrainTumour.tar -C data
```

Both approaches produce `data/Task01_BrainTumour/` with `imagesTr/` and `labelsTr/`; the archive also has metadata and unlabeled `imagesTs/`. This project downloads no data implicitly beyond the explicit subset command. The 20-case subset is split by patient for training, validation, and held-out test; official `imagesTs` cases do not include labels and cannot be scored with Dice/IoU.

## 3. Explore and preprocess

```powershell
python scripts/explore_data.py --data data/Task01_BrainTumour
```

The script reports the number of paired cases, inspects a few volume shapes and tumor voxels, and saves an example slice to `outputs/data_overview.png`. Preprocessing is deliberately minimal and reproducible: each MRI channel is z-scored over nonzero brain voxels, clipped to `[-5, 5]`, and every nonzero label is combined into one tumor mask. The data loader uses every tumor-containing axial slice plus every fourth empty slice, reducing the easy-background imbalance.

## 4. Train

```powershell
python scripts/train.py --data data/Task01_BrainTumour --max-cases 0 --epochs 20
```

Options include `--batch-size`, `--lr`, `--seed`, `--workers`, and `--output`. Set `--max-cases 0` to use every locally downloaded patient. The random split is performed on complete patient volumes before slices are generated, so slices from one patient cannot cross the train, validation, and test boundaries. The seed and arguments are recorded in the best checkpoint. Validation Dice selects the saved checkpoint; final held-out test Dice and IoU are written to `outputs/run/test_metrics.json` after training.

The objective is binary cross-entropy with logits plus soft Dice loss. Reported Dice and IoU are thresholded at 0.5 and averaged over sampled axial slices. Empty-reference/empty-prediction slices score 1, so also state that convention when comparing results. A small subset or short run may perform poorly; results must be taken from your own run and should not be presented as clinical performance.

## 5. Visualize predictions

```powershell
python scripts/predict.py --data data/Task01_BrainTumour --checkpoint outputs/run/best_model.pt
```

This saves `outputs/prediction.png` showing the FLAIR slice, its reference mask, and the model prediction. For a different case, pass `--case-index N`. The visualization chooses the axial slice with the largest reference tumor area; it is an illustrative labeled case, not a substitute for evaluating the held-out patient test set.

## 6. NeuroMask Lab: run the Streamlit app locally

Install the updated requirements, then launch the app:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

In the app, upload `outputs/run/best_model.pt` and a four-channel `.nii` or `.nii.gz` MRI volume. Choose which MRI channel to display, move through axial slices, adjust the probability threshold, and run segmentation. The app explains the preprocessing, U-Net, and thresholding steps as you use it. It displays the selected MRI sequence, predicted whole-tumor mask, and probability map. Optionally upload the matching file from `labelsTr/` to compare the selected slice with ground truth and calculate slice Dice/IoU. It expects registered, skull-stripped BraTS-style volumes with channels ordered FLAIR, T1w, T1gd, T2w; it is not a general-purpose reader for arbitrary MRI scans.

## 7. Deploy on Streamlit Community Cloud

Community Cloud deploys from a GitHub repository. Push the project code, `requirements.txt`, and this README to a repository you control. Keep `data/`, patient scans, and `*.pt` checkpoints out of Git; the `.gitignore` already excludes them. In Community Cloud, create an app from that repository and set the entrypoint to `app.py`. The current app asks the viewer to upload a checkpoint and scan at runtime, so you do not have to commit model weights or MRI files. The app viewer needs a copy of `best_model.pt` to use the deployed model.

See the official [Community Cloud deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies). Community Cloud uses `requirements.txt` to install Python packages; keep only one dependency manifest for the app.

## Metrics and interpretation

For a binary mask, Dice is `2 × intersection / (prediction pixels + reference pixels)` and IoU is `intersection / union`. Both range from 0 to 1, with larger values indicating more overlap. This implementation reports the unweighted average of per-slice metrics. For a stronger portfolio study, add per-patient volume metrics, repeat several seeds, compare against a simple baseline, and discuss the effects of 2D context and class imbalance.

## Example run

One preliminary run on the 20-case subset used 5 epochs on CPU. The seeded patient-level split contained 14 training cases, 3 validation cases, and 3 held-out test cases. At epoch 5, validation Dice was `0.6031` and validation IoU was `0.5379`. The final held-out test metrics reported by that run were Dice `0.7279` and IoU `0.6571`.

These scores are specific to this small run. Metrics are averaged over sampled axial slices, and empty-reference/empty-prediction slices score 1; the three-patient test set is too small for a reliable benchmark claim. The prediction image below is a qualitative example from `BRATS_004`, not a confirmed held-out test case.

![FLAIR input, reference mask, and U-Net prediction for BRATS_004](outputs/prediction.png)

The example prediction follows the main tumor region and includes a few small isolated false-positive areas. It illustrates one slice only and does not replace test-set evaluation.

## Limitations

This is an educational research prototype. It produces a 2D binary mask, does not preserve the separate tumor subregions, does not exploit 3D context, and is not validated for diagnosis or treatment. The 20-case subset is intentionally small. A fair headline result requires a clearly documented split, complete training details, and repeated runs; these preliminary metrics are not a robust benchmark or clinical performance claim.

## License

The source code in this repository is released under the [MIT License](LICENSE). This license applies to the project code only. The Medical Segmentation Decathlon BrainTumour data is **not** included in this repository and remains under its separate **CC BY-SA 4.0** terms. Follow the dataset's attribution and share-alike requirements when using or sharing dataset-derived material; the MIT license does not replace or override those terms.
