import yaml
import os
from robotarium_gym.utilities.misc import run_env, objectview
import argparse


def main():
    module_dir = os.path.dirname(__file__)
    if module_dir.split("/")[-1] != "robotarium_gym":
        module_dir = ""
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='PredatorCapturePrey', help='scenario name')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save the images. Show fig must be true. Set to None not to save')
    parser.add_argument('--config-dir', type=str, default=None, help='Directory to configuration file. Default (None) will be in the scenarios directory.')
    args = parser.parse_args()

    if module_dir == "":
        config_path = "config.yaml"
    elif args.config_dir is not None:
        config_path = os.path.join(module_dir, "scenarios", args.scenario,args.config_dir, "config.yaml")
    else:
        config_path = os.path.join(module_dir, "scenarios", args.scenario, "test_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = objectview(config)

    run_env(config, module_dir, args.save_dir, eval_dir=config_path)

if __name__ == '__main__':
    main()