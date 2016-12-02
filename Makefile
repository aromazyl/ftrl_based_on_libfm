all: predictor
	cd src/libfm; make all

libFM:
	cd src/libfm; make libFM

predictor:
	cd src/fm_core; make

clean:
	cd src/libfm; make clean



