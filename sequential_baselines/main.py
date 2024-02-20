import argparse

from baseline import run_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--project_name", type=str, default='default_project')
    parser.add_argument("--run_name", type=str, default='default_run')
    args = parser.parse_args()
    run_all(args.config, args.project_name, args.run_name)
