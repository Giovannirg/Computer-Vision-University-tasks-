to create a Training environment:
First, create a new empty folder and a virtual python environment venv from the command prompt.
https://docs.python.org/3/library/venv.html
Create an empty Folder and then Activate venv (shown for Windows, similar for other OS):
lab> .\venv\Scripts\activate
The prompt will change, indicating the venv name. VS Code (and maybe other IDEs) will recognize
the venv and ask for using it, confirm.
All pip installations will now be local to the venv. You may later deactivate venv by entering
deactivate and re-activate it by executing the activate script again.
First upgrade pip itself such that it finds the current versions of packages:
(venv) lab> python -m pip install –upgrade pip
Get the list of requirements from moodle and install them in your venv:
(venv) lab> python -m pip install -r requirements.txt
For the first time, this may take a while and will hopefully not fail. The exact version of Tensorflow is
not very important, the code works with Tensorflow 2.2 and newer. For further help, see
https://www.tensorflow.org/install/pip
create a new python file cifar_cnn_training.py to be filled with the training code.
