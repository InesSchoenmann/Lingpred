# Lingpred
Code and preprocessed data used for the analysis in the following journal article: 

Schönmann, I., Szewczyk, J., de Lange, F. P., & Heilbron, M. (2025). Stimulus dependencies—rather than next-word prediction—can explain pre-onset brain encoding during natural listening. _ELife_.


**For access to the neural, text and audio data used, please refer to the original publications:**
- Armeni K, Güçlü U, van Gerven M, Schoffelen JM. A 10-hour within-participant magnetoencephalography narrative dataset to test models of language comprehension. _Scientific Data_. 2022; **9**(1):278.
- Gwilliams L, Flick G, Marantz A, Pylkkanen L, Poeppel D, King JR. MEG-MASC: a high-quality magneto-encephalography dataset for evaluating natural speech processing. _arXiv preprint arXiv_:220811488. 2022
- Goldstein A, Zada Z, Buchnik E, Schain M, Price A, Aubrey B, Nastase SA, Feder A, Emanuel D, Cohen A, et al. Shared computational principles for language processing in humans and deep language models. _Nature Neuroscience_. 2022; 25(3):369–380


### Installing the environments: 

```bash 
conda env create --file main_environment.yml --name main_analysis_env
conda env create --file audio_environment.yml --name audio_env
```
