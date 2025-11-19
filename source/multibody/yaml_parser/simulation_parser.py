from pathlib import Path
import yaml


def add_simulation_info(path: str | Path, YAMLclass):
    """One-liner wrapper:  ex = load_yaml_as_example("myfile.yaml")"""

    raw = yaml.safe_load(Path(path).read_text())['simulation']

    # process the raw data
    time_step   = raw.get('time_step')
    tspan       = raw.get('end_time')
    gravity     = raw.get('gravity',9.81)  # default gravity

    YAMLclass.TimeStep  = time_step
    YAMLclass.tspan     = tspan
    YAMLclass.g         = abs(sum(gravity))  # sum in case gravity is a vector
