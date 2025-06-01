# 🎹 InsRec: Musical Instrument Recognition App

Using state-of-the-art 📈 time series analysis neural networks for musical instrument recognition!

🚀 Powered by [PyOmniTS](https://github.com/Ladbaby/PyOmniTS), the unified framework for time series analysis.

![](images/screenshot_MIC.png)

## Installation

### From Source

1. Clone this repository and its submodules, then checkout to branch `MIC` for backend submodule.

    ```shell
    git clone --recurse-submodules https://github.com/Ladbaby/InsRec
    cd InsRec/backend
    git checkout MIC
    cd ..
    ```

2. Create a python virtual environment via the tool of your choice

    for example, using [Miniconda](https://docs.conda.io/en/latest/miniconda.html)/[Anaconda](https://www.anaconda.com/):

    ```shell
    conda create -n InsRec python=3.11
    conda activate InsRec
    ```

    > Python 3.11 has been tested. Other versions may also work.

3. Install dependencies in the created environment

    ```shell
    pip install -r backend/requirements.txt
    pip install -r requirements.txt
    ```

## Usage

### Easy: Use Existing Model Weights

The web UI is launched via:

```shell
streamlit run main.py
```

or running `sh main.sh`.

### Advanced: Train a Model

Neural network training is powered by [PyOmniTS](https://github.com/Ladbaby/PyOmniTS) framework.

#### Obtain OpenMIC Dataset

Download the dataset from [here](https://zenodo.org/records/1432913), and place the extracted result under `backend/storage/datasets/OpenMIC`.
Create the parent folder if not exists.

#### Train the Model

The training procedure for new model is still a little bit complex now, later updates will try to simply it.

In `backend/models/_OpenMIC_Adaptor.py`:

```python
model_module = importlib.import_module("models." + "Linear")
```

Change "Linear" to the model name you want, where available options can be found in `backend/models`

You may find reference experimental settings (e.g., learning rate, d_model) for the chosen model in its scripts under `backend/scripts/CHOSEN_MODEL`.

Start training by:

```shell
cd backend
sh scripts/CHOSEN_MODEL/OpenMIC.sh
```

Model weights `pytorch_model.bin` will be found under `backend/storage/results`

To load your trained model, place the `pytorch_model.bin` file under `backend/storage/pretrained/OpenMIC/CHOSEN_MODEL` folder.

## Model Performance Benchmark

Test set performance on OpenMIC dataset:

|Model|Accuracy|Precision|Recall|F1
|---|---|---|---|---|
|Pyraformer|67.66|62.70|59.46|59.85
|Reformer|67.34|63.73|59.21|59.91
|Nonstationary Transformer|67.03|62.77|59.12|59.51
|TimesNet|65.83|60.28|57.14|57.21
|Mamba|65.78|61.07|56.46|57.5
|TSMixer|65.63|61.83|55.91|57.55
|LightTS|65.26|61.09|54.53|56.58
|Transformer|65.10|66.46|56.20|57.74
|DLinear|64.48|61.67|54.67|56.36
|Linear|64.32|62.79|52.85|55.69
|GRU-D|64.32|58.75|54.91|55.08
|mTAN|60.89|53.87|44.75|46.73
|Autoformer|54.48|46.35|44.40|44.68