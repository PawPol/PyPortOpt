# PyPortOpt

## How to Install?

First create a virtual environment using venv module

```bash
mkdir env && python3 -m venv ./env && source ./env/bin/activate
```
Then install using:

```bash
pip install .
```

## How to use it in your scripts?
```python
from PyPortOpt import Optimizers as o
print(o.testFunction())
True
```