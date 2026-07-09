# Lingpred

Code and preprocessed data used for the analysis in the following journal article:

Schönmann, I., Szewczyk, J., de Lange, F. P., & Heilbron, M. (2025). Stimulus dependencies—rather than next-word prediction—can explain pre-onset brain encoding during natural listening. _ELife_.

**For access to the neural, text and audio data used, please refer to the original publications:**
- Armeni K, Güçlü U, van Gerven M, Schoffelen JM. A 10-hour within-participant magnetoencephalography narrative dataset to test models of language comprehension. _Scientific Data_. 2022; **9**(1):278.
- Gwilliams L, Flick G, Marantz A, Pylkkanen L, Poeppel D, King JR. MEG-MASC: a high-quality magneto-encephalography dataset for evaluating natural speech processing. _arXiv preprint arXiv_:220811488. 2022
- Goldstein A, Zada Z, Buchnik E, Schain M, Price A, Aubrey B, Nastase SA, Feder A, Emanuel D, Cohen A, et al. Shared computational principles for language processing in humans and deep language models. _Nature Neuroscience_. 2022; 25(3):369–380

## Repository structure

- [`lingpred_new/`](lingpred_new) — core analysis package: encoding models (`encoding_analysis.py`), MEG preprocessing (`preprocessing.py`), data I/O (`io.py`), plotting (`plotting.py`), and shared helpers (`utils.py`).
- [`lingpred_audio/`](lingpred_audio) — acoustic feature extraction (mel-spectrogram averaging per word) used for the acoustic control models.
- [`gpt2/`](gpt2) — wrapper around GPT-2 (via HuggingFace `transformers`) used to extract word-level GPT-2 embeddings and surprisal.
- [`notebooks/`](notebooks) — analysis notebooks, see below.
- [`audio/`](audio) — per-dataset acoustic features and word-event files (Armeni, Gwilliams, Goldstein).
- [`Goldstein_gpt_features/`](Goldstein_gpt_features) — GPT features/transcript data for the Goldstein podcast dataset.
- [`figures/`](figures) — main and supplementary figures from the paper.
- [`main_environment.yml`](main_environment.yml) / [`audio_environment.yml`](audio_environment.yml) — conda environment specifications.

### Notebooks

- [`Make_Audio_X_Matrix.ipynb`](notebooks/Make_Audio_X_Matrix.ipynb) — builds the mel-spectrogram feature matrix (acoustic model) for a given dataset.
- [`Audio_Encoding.ipynb`](notebooks/Audio_Encoding.ipynb) — runs the encoding analysis using the acoustic (audio-based) model.
- [`Compute_Selfpredictability.ipynb`](notebooks/Compute_Selfpredictability.ipynb) — computes word/stimulus self-predictability measures from GPT-2, GloVe, and arbitrary vectors.
- [`Compute_Brainscore.ipynb`](notebooks/Compute_Brainscore.ipynb) — computes the encoding brainscore for the Armeni dataset across GloVe/GPT/arbitrary vectors, with optional regressed-out and bigram-removed variants.
- [`Figure.ipynb`](notebooks/Figure.ipynb) — reproduces the paper's main figures (base pre-onset encoding effects, predicted vs. not-predicted word splits).
- [`adjusting_goldstein_audio.ipynb`](notebooks/adjusting_goldstein_audio.ipynb) — utility notebook for aligning/trimming the Goldstein podcast audio to the transcript.

## Installing the environments

```bash
conda env create --file main_environment.yml --name main_analysis_env
conda env create --file audio_environment.yml --name audio_env
```
