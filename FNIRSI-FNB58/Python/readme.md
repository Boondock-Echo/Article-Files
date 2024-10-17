Please use and extend these files, feel free to fork. Support is not available at this time.  Please attribute Boondock Technologies as the original author.

I used these python files to read data from the FRN58 USB Multimeter datalogger.

To collect voltage and current data from a FRN58, please use [FNB58DataCollection_VoltageAndCurrent.py](FNB58DataCollection_VoltageAndCurrent.py). But be sure to change the Bluetooth Address to your devices bluetooth address -- which you can find using [bluetoothServicesScanner.py](bluetoothServicesScanner.py)

And because things change with time, I copied the virtual environment requirements over to [requirements.txt](requirements.txt) and you can install them (after creating and activating a new virtual environment) by typing `pip install -r requirements.txt`

Good luck!
Mark
