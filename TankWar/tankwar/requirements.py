import os
os.system("conda list -e > requirements_conda.txt")
os.system("pip list --format=freeze > requirements_pip.txt")
