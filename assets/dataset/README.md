# Dataset
Run the following code in terminal to download the dataset of all transactions that have happened in the *Coinbase* exchange till the current time:
```
chmod +x ./data_download.sh
./data_download.sh
```
This will download the dataset to `coinbaseUSD.csv` which can be further processed to generate the per minute Bitcoin prices using [this](../../code/process/generate.py).