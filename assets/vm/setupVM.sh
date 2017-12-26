#!/bin/bash
while read LINE; do sudo apt-get install "$LINE"; done < requirements.txt

#install ta-lib for ubuntu
sudo tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
sudo ./configure
sudo make
sudo make install
cd ..

#install ta-lib python wrapper
sudo tar -xzf TA-Lib-0.4.10.tar.gz
cd TA-Lib-0.4.10/
python setup.py install
cd ..

#install python dependencies
sudo pip install pip-9.0.1-py2.py3-none-any.whl
for file in ./basewhlfiles/*; do
    sudo pip install ${file##*/}
done

#install python libraries
for file in ./whlfiles/*; do
    sudo pip install ${file##*/}
done

