VENV_NAME = venv
PYTHON = $(VENV_NAME)/bin/python3.8

SRC = src/
PLOT = plot/
LOG = log/
MODEL = models/

EXE = $(SRC)main.py

EXE_ARGS = vgg1 vgg2 vgg3

.PHONY: all runAll install activate clean checkDir

all: checkDir runAll

runAll: checkDir install activate
	for arg in $(EXE_ARGS); do \
		$(PYTHON) $(EXE) --model $$arg --epoch 20 | tee $(LOG)stdout.$$arg.log \
		2> $(LOG)stderr.$$arg.log; \
		$(PYTHON) $(EXE) --model $$arg --epoch 50 --imgAgu | tee $(LOG)stdout.$$arg.imgAgu.log \
		2> $(LOG)stderr.$$arg.imgAgu.log; \
		$(PYTHON) $(EXE) --model $$arg --epoch 20 --pandas | tee $(LOG)stdout.$$arg.pandas.log \
		2> $(LOG)stderr.$$arg.pandas.log; \
		$(PYTHON) $(EXE) --model $$arg --epoch 50 --pandas --imgAgu | tee $(LOG)stdout.$$arg.pandas.imgAgu.log \
		2> $(LOG)stderr.$$arg.pandas.imgAgu.log; \
	done; \

	$(PYTHON) $(EXE) --model dropout --epoch 50 | tee $(LOG)stdout.dropout.log \
	2> $(LOG)stderr.dropout.log; \
	$(PYTHON) $(EXE) --model dropout --epoch 50 --imgAgu | tee $(LOG)stdout.dropout.imgAgu.log \
	2> $(LOG)stderr.dropout.imgAgu.log; \
	$(PYTHON) $(EXE) --model dropout --epoch 50 --pandas | tee $(LOG)stdout.dropout.pandas.log \
	2> $(LOG)stderr.dropout.pandas.log; \
	$(PYTHON) $(EXE) --model dropout --epoch 50 --imgAgu --pandas | tee $(LOG)stdout.dropout.pandas.imgAgu.log \
	2> $(LOG)stderr.dropout.pandas.imgAgu.log; \
	

install: activate $(LOG)
	$(VENV_NAME)/bin/pip install --upgrade pip | tee $(LOG)stdout.python.log 2> $(LOG)stderr.python.log
	$(VENV_NAME)/bin/pip install -r requirements.txt | tee $(LOG)stdout.python.log 2> $(LOG)stderr.python.log

activate: $(VENV_NAME)
	. $(VENV_NAME)/bin/activate

$(VENV_NAME):
	python3.8 -m venv $(VENV_NAME)

checkDir: $(LOG) $(PLOT) $(MODEL) install activate
	$(PYTHON) $(EXE) --onlyCreateDir

$(LOG) $(PLOT) $(MODEL):
	mkdir -p $@

clean:
	rm -r $(VENV_NAME)