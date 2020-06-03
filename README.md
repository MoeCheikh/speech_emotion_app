This is a speech emotion recognition model creation and application repo. It will allow you to train a modular MLP model with the RAVDESS dataset, and then use that model with a flask application to predict the speech emotion of any .wav file. 


### REQS:

To download the RAVDESS speech emotion recognition data, go to: https://drive.google.com/file/d/1wWsrN2Ep7x6lWqOXfr4rpKGYrJhWc8z7/view

for installing all dependencie simply open terminal and run:

```
. ./install_deps.sh
```

This should create your venv and populate it with all necessary dependencies

### MODEL:

A multilayer perceptron model to detect the emotion of wav files.
To create and edit the model see create_model.py
Once the create_model.py is adjusted to your liking (emotions_to_observe, and path to sound data), simply run:

```
python3 create_model.py
```
to create the model.model binary file and test accuracy of your model


### APP:

Once the model.model binary is created, you can spin up the flask application (ToneCheck):
To do so run

```
. ./start_flask.sh
```

The app will run default on localhost:5000, the emotions available for predictions will correspond with the emotions_to_observe variable you have edited inside create_models.py (and are therefore available inside the model binary file)

