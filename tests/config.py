import os
import yaml
from box import Box

file_path_list = [
    "./test_config.yml",
    "../test_config.yml",
    "../tests/test_config.yml",
    "./tests/test_config.yml"
]

file_path = None
for path in file_path_list:
    if os.path.exists(path):
        file_path = path
        break

if file_path is not None:
    with open(file_path, 'r') as yml_file:
        try:
            config = yaml.safe_load(yml_file)
        except yaml.YAMLError as exc:
            print(exc)

cfg = Box({**config["base"]},
          default_box=True,
          default_box_attr=None)
