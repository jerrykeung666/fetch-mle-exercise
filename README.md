# fetch-mle-exercise
2023 Summer Fetch Reward MLE intern exercise

## Envirionment
To create the conda environment, please run the following command:
```
conda env create -f environment.yml
```

## Training
To train the model, please run the following command:
```
python src/train.py --path <path-to-data-daily-csv-file> --epochs <your-epochs-to-train>
```
`<path-to-data-daily-csv-file>`: the absolute path of `date_daily.csv` file on your machine.

`<your-epochs-to-train>`: training epochs, by default 1000.

## Eval
To get the curve of "number of scripts - each day in 2022" or "number of scripts - each month in 2022", please run:
```
python src/eval.py --model-path <path-to-trained-model>
```
`<path-to-trained-model>`: the absolute path of `estimated_number_day.pth` file on your machine. (Should be in the `results/` directory).

The results can be found under the `results/` directory.