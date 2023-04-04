# HMTISC: Human Mobility Prediction based on Trend Iteration of Spectral Clustering

Codes: https://github.com/wenzhenJia/HMTISC


## Environment setting
```bash
conda create -n python36 python=3.6
pip install keras==1.2.1
pip install sklearn
conda install theano pygpu h5py pandas seaborn scikit-learn
```

`Keras 1.2 + Theano` configuration file of Keras: `~/.keras/keras.json`

```json
{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```

## Path setting
```
# Windows
set DATAPATH=[path_to_your_data]

# Linux
export DATAPATH=[path_to_your_data]
```

## File introduction
```md
mian.py: configure the number of clusters and the number of iterations, select the prediction model
nyc.txt: Longitude and latitude of bike-sharing stations in New York
generate_spectral.py: clustering stations using spectral clustering
generate_dataset.py: generate a two-channel matrix
ST3DNet: ST3DNet prediction model
STResNet: STResNet prediction model
```

## Prepare data
Datasets: BikeNYC and BikeDC are the datasets we used in the paper, it suffices to reproduce the results what we have reported in the paper.

Download BikeNYC dataset provided by ./NYCdata/ in https://drive.google.com/file/d/1OQXaE6g4EsuMdISiHroerUfpXqTWWKOk/view?usp=sharing

Download BikeDC dataset provided by ./DCdata/ in https://drive.google.com/file/d/1OQXaE6g4EsuMdISiHroerUfpXqTWWKOk/view?usp=sharing

## Run the model
```bash
THEANO_FLAGS="device=cuda0,floatX=float32" python main.py 
```