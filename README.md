# Plant Disease Diagnosis

Starter structure for exploring plant-disease images, training CNN and ViT
classifiers, and inspecting the domain gap between datasets.

## Project layout

- `configs/config.yaml`: shared dataset and training settings.
- `notebooks/`: data exploration, CNN, ViT, and domain-gap workflows.
- `src/`: reusable dataset, model, training, and evaluation code.
- `app/`: Streamlit image-upload interface.

## Setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Arrange an image-folder dataset like this (class names are directory names):

```text
data/
  train/
    healthy/
    diseased/
  val/
    healthy/
    diseased/
  test/
    healthy/
    diseased/
```

Update the paths and training settings in `configs/config.yaml` to match your
dataset. The notebooks are starting points; training and inference workflows
should be filled in for the selected dataset and experiment.

Run the app with:

```bash
streamlit run app/streamlit_app.py
```

## Notebooks

1. `01_data_exploration.ipynb` — inspect labels, class balance, and sample images.
2. `02_train_cnn.ipynb` — develop a CNN baseline.
3. `03_train_vit.ipynb` — develop a Vision Transformer experiment.
4. `04_domain_gap.ipynb` — compare dataset distributions and model performance.
# plant-disease-diagnosis
