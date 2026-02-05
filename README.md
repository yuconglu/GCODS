# GCODS

## 1. Environment Setup

The project is developed with Python 3.11 and PyTorch 2.4.0 (for CUDA 12.8).

The main dependencies are:
- numpy==1.26.4
- pandas==2.2.2
- torch==2.4.0
- torchvision==0.19.0
- torch-geometric==2.5.3
- torchdiffeq==0.2.3
- optuna==3.6.1
- cliffordlayers==1.3.1
- timm==0.9.12


## 2. Dataset Download

The model is trained and evaluated on the ERA5 reanalysis dataset, which is publicly available.

You can access and download the ERA5 data from the ECMWF Climate Data Store:
[**WeatherBench Dataset**](https://mediatum.ub.tum.de/1524895)

## 3. How to Run

The model training and evaluation are handled by the `Run.py` script. All hyperparameters and settings are managed through configuration files located in the `GCODS-main/model/` directory.

### Training

To start a training session, run the `Run.py` script. The script will automatically use the `Global.conf` file by default.

```bash
cd GCODS-main/model
python Run.py --mode train
```

The best performing model based on the validation set will be saved automatically to the `GCODS-main/model/trained-best-model/` directory.

### Evaluation

To evaluate a trained model on the test set, use the `test` mode. This will load the saved best model and compute the final performance metrics.

```bash
cd GCODS-main/model
python Run.py --mode test
```

### Hyperparameter Optimization

The framework supports hyperparameter tuning using Optuna. You will need to create a separate Python script to define the search space and run the optimization study by repeatedly calling the `run_training_job` function from `Run.py`.

