# Honors Thesis:

## Time Series Analysis of Chaotic Dynamical Systems with Recurrent Neural Networks

## Directory Layout

* Source code for my final project is stored in `src` and `proj`.
* `src` contains the package I built to assist with the experimentation
* `proj` contains the testing and experimentation I did to generate the results
summarized in the report.
* `MANIFEST.in`, `Makefile`, and `setup.py` are used for the package.

## Installing `echonn`

```
git clone https://github.com/larkwt96/honors-thesis.git echonn
cd echonn
python3 -m pip install .
```

or without cloning

```
python3 -m pip install git+git://github.com/larkwt96/honors-thesis.git
```

## Testing

```
python3 -m unittest discover -s src
```

See the Makefile for aliases.
