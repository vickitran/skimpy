# skimpy
Summary statistics prettier. Heavily inspired by [skimr](https://github.com/ropenscilabs/skimr)

# how to use

``` python
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import skimpy.main as skimpy

data = load_iris()
toy_data = pd.DataFrame(data['data'],columns = data["feature_names"])
toy_data["target"] = data.target

data_dict = pd.DataFrame([0,1,2],columns = ["target"])
data_dict["target_names"] = data.target_names

toy_data = pd.merge(toy_data,data_dict,how = "left", on = "target")

skimpy.skim(toy_data.drop("target",1),["target_names"])

```
# output
![console-output](https://github.com/vickitran/skimpy/blob/master/skimpy/ex_img/ex_img.png)

# warning!!!
skimpy does not play nice on windows computers. [unicode block characters - windows support](https://stackoverflow.com/questions/28485545/where-is-upper-one-quarter-block-letter-in-unicode)
