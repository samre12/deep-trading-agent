# Change Log

- removed an import of `talib` from `processor.py` that caused error due to inappropriate dependencies

- added support for both `VALID` and `SAME` padding types for the convolutional layer (can be passed as `VALID/SAME` in the configuration file)