import yaml
from box import Box

with open("test_config.yml", 'r') as yml_file:
    try:
        config = yaml.safe_load(yml_file)
    except yaml.YAMLError as exc:
        print(exc)

cfg = Box({**config["base"]},
          default_box=True,
          default_box_attr=None)