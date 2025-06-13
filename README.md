# 🎹 InsRec: Musical Instrument Recognition App

Using state-of-the-art 📈 time series analysis neural networks for musical instrument recognition!

🚀 Powered by [PyOmniTS](https://github.com/Ladbaby/PyOmniTS), the unified framework for time series analysis.

> [!IMPORTANT]
> Accuracy is not guaranteed (and I'm not an expert in music)! Refer to the benchmark section for model performance details.

## 📷 Screenshot

![](images/screenshot_MIC.png)

## 🌟 Features

Models are currently trained on the [OpenMIC-2018 dataset](https://zenodo.org/records/1432913), which includes 20 types of "instruments":

0. 🪗 Accordion [[wiki]](https://en.wikipedia.org/wiki/Accordion)
1. 🪕 Banjo [[wiki]](https://en.wikipedia.org/wiki/Banjo)
2. Bass [[wiki]](https://en.wikipedia.org/wiki/Bass_(sound))
3. Cello [[wiki]](https://en.wikipedia.org/wiki/Cello)
4. Clarinet [[wiki]](https://en.wikipedia.org/wiki/Clarinet)
5. Cymbals [[wiki]](https://en.wikipedia.org/wiki/Cymbals)
6. 🥁 Drums [[wiki]](https://en.wikipedia.org/wiki/Drum)
7. Flute [[wiki]](https://en.wikipedia.org/wiki/Flute)
8. 🎸 Guitar [[wiki]](https://en.wikipedia.org/wiki/Guitar)
9. Mallet Percussion [[wiki]](https://en.wikipedia.org/wiki/Keyboard_percussion_instrument)
10. Mandolin [[wiki]](https://en.wikipedia.org/wiki/Mandolin)
11. Organ [[wiki]](https://en.wikipedia.org/wiki/Organ_(music))
12. 🎹 Piano [[wiki]](https://en.wikipedia.org/wiki/Piano)
13. 🎷 Saxophone [[wiki]](https://en.wikipedia.org/wiki/Saxophone)
14. Synthesizer [[wiki]](https://en.wikipedia.org/wiki/Synthesizer)
15. Trombone [[wiki]](https://en.wikipedia.org/wiki/Trombone)
16. 🎺 Trumpet [[wiki]](https://en.wikipedia.org/wiki/Trumpet)
17. Ukulele [[wiki]](https://en.wikipedia.org/wiki/Ukulele)
18. 🎻 Violin [[wiki]](https://en.wikipedia.org/wiki/Violin)
19. 🗣️ Voice [[wiki]](https://en.wikipedia.org/wiki/Human_voice)

## ⏬ Installation

### From Source

1. Clone this repository and its submodules, then checkout to branch `InsRec` for backend submodule.jh

    ```shell
    git clone --recurse-submodules https://github.com/Ladbaby/InsRec
    cd InsRec/backend
    git checkout InsRec
    cd ..
    ```

2. Create a python virtual environment via the tool of your choice.

    for example, using [Miniconda](https://docs.conda.io/en/latest/miniconda.html)/[Anaconda](https://www.anaconda.com/):

    ```shell
    conda create -n InsRec python=3.12
    conda activate InsRec
    ```

    > Python 3.11 & 3.12 have been tested. Other versions may also work.

3. Install dependencies in the created environment.

    ```shell
    pip install -r backend/requirements.txt
    pip install -r requirements.txt
    ```

    > Some models may require extra dependencies, which can be found in comments of `backend/requirements.txt`.

## 🚀 Usage

### Easy: Use Existing Model Weights

The web UI is launched via:

```shell
streamlit run main.py
```

or running `sh main.sh`.

During the first run, it will prompt you whether to download checkpoint files for models in the terminal.

### Advanced: Train a Model

Neural network training is powered by [PyOmniTS](https://github.com/Ladbaby/PyOmniTS) framework.

The training procedure for existing models on OpenMIC-2018 dataset is detailed here.

#### Obtain OpenMIC Dataset

- Download the dataset from [here](https://zenodo.org/records/1432913), and place the extracted result under `backend/storage/datasets/OpenMIC`.
Create the parent folder if not exists.
- Download the processed VGGish representations of corresponding audios from [huggingface](https://huggingface.co/datasets/Ladbaby/InsRec-datasets/blob/main/OpenMIC/processed/x_repr_times.npy), and place it under `backend/storage/datasets/OpenMIC/processed`.

    > It's worth noting that these VGGish representations are different from the "X" in `backend/storage/datasets/OpenMIC/openmic-2018.npz`. Our representations are obtained using the pretrained [PyTorch VGGish pipeline](https://docs.pytorch.org/audio/master/generated/torchaudio.prototype.pipelines.VGGISH.html) and the PCA weights from [torchvggish](https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish_pca_params-970ea276.pth).

#### Train the Model

You may find experimental settings (e.g., learning rate, d_model) for the chosen model in its scripts under `backend/scripts/CHOSEN_MODEL/OpenMIC.sh`.

Start training by:

```shell
cd backend
sh scripts/CHOSEN_MODEL/OpenMIC.sh
```

Model weights `pytorch_model.bin` will be found under `backend/storage/results`

To infer using your trained weights instead, replace the `pytorch_model.bin` file under `backend/storage/pretrained/OpenMIC/CHOSEN_MODEL` folder with your own.

## 📊 Model Performance Benchmark

Test set performance on OpenMIC-2018 dataset:

|Model|Accuracy|Precision|Recall|F1
|---|---|---|---|---|
|Pyraformer|67.71|64.74|64.95|64.18
|Reformer|67.50|64.45|64.30|63.65
|Informer|66.98|63.41|63.74|62.73
|Nonstationary Transformer|66.46|63.88|63.06|62.66
|Hi-Patch|65.90|63.72|60.76|61.12
|GRU-D|65.83|63.30|62.34|61.95
|TimesNet|65.52|62.74|61.58|61.41
|Mamba|65.26|61.97|61.25|60.85
|TSMixer|65.21|62.35|60.59|60.64
|Raindrop|65.16|62.30|62.33|61.29
|LightTS|65.05|63.32|60.34|60.95
|Transformer|65.00|63.12|63.71|61.87
|FEDformer|64.48|60.58|59.96|59.61
|FreTS|64.48|62.12|59.30|59.99
|DLinear|64.22|62.04|59.04|59.64
|Linear|64.17|63.07|58.41|59.60
|Leddam|63.18|59.77|59.97|58.52
|iTransformer|63.18|60.99|57.83|58.44
|PrimeNet|62.19|57.97|57.53|56.76
|mTAN|60.89|53.87|44.75|46.73
|SegRNN|58.96|61.84|50.58|53.43
|Autoformer|54.43|52.15|50.25|50.56
|PatchTST|42.97|43.57|37.50|37.59
|MICN|36.61|33.68|29.31|29.54
|SeFT|35.99|28.39|25.02|24.91
|TiDE|34.58|30.69|30.42|30.21
|Crossformer|21.72|1.09|5.00|1.78
|FiLM|21.72|1.09|5.00|1.78


Existing state-of-the-art time series models mainly learns in the time domain, while audios processing models primarily learns in the frequency domain. 
Also, audio (e.g., 16k every second) is far longer than any time series in research datasets (e.g., 720).
Therefore, [VGGish](https://docs.pytorch.org/audio/master/generated/torchaudio.prototype.pipelines.VGGISH.html) is currently used as the encoder to convert audio input as embeddings, and time series models take them as input instead (it makes little sense I know, but this is possibly the only way for painless adaptation).

Further improvements may require changing network architecture of time series models, such that VGGish embeddings are treated as representations instead of time series.
