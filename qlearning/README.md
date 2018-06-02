For testing:
```
# create conda env
conda create -n tf-gym-1.1 python=3.6
source activate tf-gym-1.1

# install dependencies
pip install -r requirements.txt
pip install gym[atari] # not installing automatically from requirements.txt

# so we can load our conda env in jupyter
python -m ipykernel install --user --name=tf-gym-1.1

jupyter notebook
```
