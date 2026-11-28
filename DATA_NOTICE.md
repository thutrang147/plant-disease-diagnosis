# Data notice

The MIT licence in `LICENSE` covers code and documentation only. Images and datasets keep their own terms.

| Data | Source used | Licence / status | What this repository contains |
| --- | --- | --- | --- |
| PlantVillage (lab) | Hugging Face mirror `geraldmc/plantvillage-full`, pinned to revision `f808f27` (the default branch used for the splits and evaluations), of Hughes & Salathé (2015) | **Sources disagree:** most mirrors state CC0 1.0. The original release accompanying arXiv:1511.08060 is described as CC BY-SA 3.0, including a share-alike condition on trained models. Treat as CC BY-SA 3.0 (attribution + share-alike) until confirmed. | Row-index manifests (`data/splits/lab_*.csv`), a few example images embedded in Notebooks 01 and 05 |
| PlantDoc (field) | Hugging Face mirror `geraldmc/plantdoc-full`, revision `v0.1.0`, of Singh et al. (2020) | CC BY 4.0 (attribution) per the original repository. Many PlantDoc images are web photos carrying stock-agency watermarks; the dataset licence does not grant rights in those underlying photographs. | Row-index manifests (`data/splits/field_*.csv`), example images embedded in Notebooks 01 and 10 and in `outputs/error_analysis_development/*.png` |
| External v1 / v2 online disease images (46) | Extension and university web pages recorded in the private manifest | `redistribution_rights = not_established`. **Not distributed.** | Nothing in the current tree. They exist in Git history from commit `59e3bc4`; see below. |
| Self-collected Healthy photographs and crops | Project team | Owned by the team; may contain EXIF metadata | Not distributed in the current tree (also present in history from `59e3bc4`) |
| PlantSeg (External v3) | Third-party Hugging Face mirror `LFTTim/plantseg_disease_segmentation`, revision `2fff5d6`, of Wei et al. (2024), arXiv:2409.04038 | CC BY-NC-ND 4.0 (non-commercial, no derivatives). Web-collected images with expert-reviewed masks. | Row indices, hashes and derived statistics only (`data/external_v3_plantseg/`); no images or masks |
| Model checkpoints | Trained by the project | Derived from the datasets above. If the PlantVillage CC BY-SA 3.0 terms apply, released checkpoints would inherit share-alike conditions. | Not in Git. Shared through the project's public Google Drive folder; IDs and SHA-256 values are in `data/checkpoints.json` (`scripts/download_checkpoints.py`). Confirm the PlantVillage licence before any wider release. |

## Before making the repository public

1. Git history still contains the 62 External v1 photographs added in commit `59e3bc4`. For a public or
   submitted copy, either share a fresh snapshot without history (for example `git archive HEAD`), or have all
   collaborators agree on a history rewrite (`git filter-repo`) before publishing.
2. Decide whether the notebook galleries with watermarked PlantDoc images should stay embedded. Clearing
   those outputs does not change any reported number.
3. Confirm the PlantVillage licence on the original data page before releasing checkpoints.

## Citation of datasets

- Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv:1511.08060.
- Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc: A dataset for visual plant disease detection. CoDS-COMAD 2020.
