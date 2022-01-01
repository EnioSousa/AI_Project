VENV_NAME = venv
PYTHON = $(VENV_NAME)/bin/python3.8

SRC = src/
PLOT = plot/
LOG = log/
MODEL = models/

EXE = $(SRC)main.py

.PHONY: all runDefault runDefaultAgu runPanda runPandaAgu runAll install activate clean checkData predict 

all: checkDir runAll

run: install activate
	$(PYTHON) $(EXE) --log --generate --model vgg16 --epoch 20

# If you need help, you can see the program usage
help:
	$(PYTHON) $(EXE) --help

# run default data set (dogs, cats) without imgage agumentation
runDefault: install activate
	$(PYTHON) $(EXE) --log --generate --model vgg1 --epoch 20
	$(PYTHON) $(EXE) --log --generate --model vgg2 --epoch 20
	$(PYTHON) $(EXE) --log --generate --model vgg3 --epoch 20
	$(PYTHON) $(EXE) --log --generate --model vgg16 --epoch 20
	$(PYTHON) $(EXE) --log --generate --model dropout --epoch 20

# run default data set (dogs, cats) with image agumentation
runDefaultAgu: install activate
	$(PYTHON) $(EXE) --log --generate --model vgg1 --imgAgu --epoch 20
	$(PYTHON) $(EXE) --log --generate --model vgg2 --imgAgu --epoch 20
	$(PYTHON) $(EXE) --log --generate --model vgg3 --imgAgu --epoch 20
	$(PYTHON) $(EXE) --log --generate --model dropout --imgAgu --epoch 20
		
# run dataset pandas without img agu
runPanda: install activate 
	$(PYTHON) $(EXE) --log --generate --model vgg1 --pandas --epoch 20
	$(PYTHON) $(EXE) --log --generate --model vgg2 --pandas --epoch 20
	$(PYTHON) $(EXE) --log --generate --model vgg3 --pandas --epoch 20
	$(PYTHON) $(EXE) --log --generate --model vgg16 --pandas --epoch 20
	$(PYTHON) $(EXE) --log --generate --model dropout --pandas --epoch 20

# run dataset pandas with img agu
runPandaAgu: install activate
	$(PYTHON) $(EXE) --log --generate --model vgg1 --pandas --imgAgu --epoch 20
	$(PYTHON) $(EXE) --log --generate --model vgg2 --pandas --imgAgu --epoch 20
	$(PYTHON) $(EXE) --log --generate --model vgg3 --pandas --imgAgu --epoch 20
	$(PYTHON) $(EXE) --log --generate --model dropout --pandas --imgAgu --epoch 20
	
# run dataset pandas with resnet50 model
runResNet50: install activate
	$(PYTHON) $(EXE) --log --generate --model resNet50 --epoch 20
	$(PYTHON) $(EXE) --log --generate --model resNet50 --pandas --epoch 20
	
# Run everything
runAll: runDefault runDefaultAgu runPanda runPandasAgu
	
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
