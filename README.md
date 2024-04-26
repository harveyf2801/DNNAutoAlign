# Differentiable All-Pass DNNs

This project aims to align an input signal with a target / reference signal using differentiable all-pass filters.
The Parameter Network will control the individual parameters to align the phase of both signals. The filters consist of 2nd order all-pass filters, and are applied using the frequency sampling method to approximate a cascade of IIR filters. This method along with the main network architecture was taken from the [DASP library](https://github.com/csteinmetz1/dasp-pytorch/tree/main). A Temporal Convolutional Network (TCN) is used to train the model as it allows the model to learn based on the whole time domain signal and avoids phase estimation.

## Citations

Differentiable All-Pass Filters
```bibtex
@inproceedings{bargum2023,
author = {Bargum, Anders and Serafin, Stefania and Erkut, Cumhur and Parker, Julian},
year = {2023},
month = {06},
pages = {},
title = {Differentiable Allpass Filters for Phase Response Estimation and Automatic Signal Alignment}
}
```

MR-STFT Loss
```bibtex
@inproceedings{steinmetz2020auraloss,
    title={auraloss: {A}udio focused loss functions in {PyTorch}},
    author={Steinmetz, Christian J. and Reiss, Joshua D.},
    booktitle={Digital Music Research Network One-day Workshop (DMRN+15)},
    year={2020}
}
```

```bibtex
@article{Arik_2019,
   title={Fast Spectrogram Inversion Using Multi-Head Convolutional Neural Networks},
   volume={26},
   ISSN={1558-2361},
   url={http://dx.doi.org/10.1109/LSP.2018.2880284},
   DOI={10.1109/lsp.2018.2880284},
   number={1},
   journal={IEEE Signal Processing Letters},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Arik, Sercan O. and Jun, Heewoo and Diamos, Gregory},
   year={2019},
   month=jan, pages={94–98} }
```

Differentiable parametric EQ and dynamic range compressor
```bibtex
@article{steinmetz2022style,
  title={Style transfer of audio effects with differentiable signal processing},
  author={Steinmetz, Christian J and Bryan, Nicholas J and Reiss, Joshua D},
  journal={arXiv preprint arXiv:2207.08759},
  year={2022}
}
```

Differentiable IIR filters
```bibtex
@inproceedings{nercessian2020neural,
  title={Neural parametric equalizer matching using differentiable biquads},
  author={Nercessian, Shahan},
  booktitle={DAFx},
  year={2020}
}
```

```bibtex
@inproceedings{colonel2022direct,
  title={Direct design of biquad filter cascades with deep learning 
          by sampling random polynomials},
  author={Colonel, Joseph T and Steinmetz, Christian J and 
          Michelen, Marcus and Reiss, Joshua D},
  booktitle={ICASSP},
  year={2022},
  organization={IEEE}
}
```