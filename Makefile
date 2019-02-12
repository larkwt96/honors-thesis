.PHONY: run debug test install clean build
run:
	cd src && python3 -m echonn

debug: test
	cd src && python3 -m echonn

test:
	python3 -m unittest discover -s src

install:
	python3 -m pip install .

install_updatable:
	python3 -m pip install git+git://github.com/larkwt96/honors-thesis.git

clean:
	rm -rf src/test/data/temp
