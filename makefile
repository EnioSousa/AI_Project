VENV_NAME = venv
PYTHON = $(VENV_NAME)/bin/python3.8

SRC = src/
PLOT = plot/
LOG = log/
MODEL = models/

EXE = $(SRC)main.py

.PHONY: all run runDefault runGauss runPanda runResNet50 install activate clean checkData predict 

run: install activate
	$(PYTHON) $(EXE) --generate --featureDesc --pandas

all: checkDir runAll

# If you need help, you can see the program usage
help:
	$(PYTHON) $(EXE) --help

runFeatureDesc: install activate
# Use the default descriptons dataset 
	$(PYTHON) $(EXE) --generate --featureDesc 
# Use the pandas descriptons dataset 
	$(PYTHON) $(EXE) --generate --featureDesc --pandas


runDefault: install activate
# vgg(1|2|3|16)
	$(PYTHON) $(EXE) --log --generate --model vgg1 --epoch 20
	$(PYTHON) $(EXE) --log --generate --model vgg2 --epoch 20
	$(PYTHON) $(EXE) --log --generate --model vgg3 --epoch 20
	$(PYTHON) $(EXE) --log --generate --model vgg16 --epoch 10
# vgg3 + imgAgu
	$(PYTHON) $(EXE) --log --generate --model vgg3 --epoch 20 --imgAGu
# vgg3 + dropout
	$(PYTHON) $(EXE) --log --generate --model dropout --epoch 20

# run default data set with gauss images
runGauss: install activate
# vgg(1|2|3|16) + gauss
	$(PYTHON) $(EXE) --log --generate --model vgg1 --epoch 20 --gauss
	$(PYTHON) $(EXE) --log --generate --model vgg2 --epoch 20 --gauss
	$(PYTHON) $(EXE) --log --generate --model vgg3 --epoch 20 --gauss
	$(PYTHON) $(EXE) --log --generate --model vgg16 --epoch 20 --gauss
# vgg3 + imgAgu + gauss
	$(PYTHON) $(EXE) --log --generate --model vgg3 --epoch 20 --imgAgu --gauss
# vgg3 + dropout + gauss	
	$(PYTHON) $(EXE) --log --generate --model dropout --epoch 20 --gauss

# Run data set with panda images
runPanda: install activate
# vgg(1|2|3|16) with pandas
	$(PYTHON) $(EXE) --log --generate --model vgg1 --pandas --epoch 20
	$(PYTHON) $(EXE) --log --generate --model vgg2 --pandas --epoch 20
	$(PYTHON) $(EXE) --log --generate --model vgg3 --pandas --epoch 20
	$(PYTHON) $(EXE) --log --generate --model vgg16 --pandas --epoch 20
# vgg3 + imgAgy with pandas
	$(PYTHON) $(EXE) --log --generate --model vgg3 --pandas --epoch 20 --imgAgu
# vgg3 + dropout with pandas
	$(PYTHON) $(EXE) --log --generate --model dropout --pandas --epoch 20

# run dataset pandas with resnet50 model
runResNet50: install activate
	$(PYTHON) $(EXE) --log --generate --model resNet50 --epoch 10
	$(PYTHON) $(EXE) --log --generate --model resNet50 --epoch 10 --gauss
	$(PYTHON) $(EXE) --log --generate --model resNet50 --pandas --epoch 10
	
# Predict
predict:
	$(PYTHON) $(EXE) --predict --model vgg1 

# instal in enviroment
install: activate $(LOG)
	$(VENV_NAME)/bin/pip install --upgrade pip 
	$(VENV_NAME)/bin/pip install -r requirements.txt 

# activate enviromenta
activate: $(VENV_NAME)
	. $(VENV_NAME)/bin/activate

# Check data set directories
checkData: $(LOG) $(PLOT) $(MODEL) install activate
	$(PYTHON) $(EXE) --checkData

# Remove enviroment isntalation
clean:
	rm -r $(VENV_NAME)

$(VENV_NAME):
	python3.8 -m venv $(VENV_NAME)

$(LOG) $(PLOT) $(MODEL):
	mkdir -p $@
