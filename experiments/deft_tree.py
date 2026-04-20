import hydra

from run_deft_tree import run_experiment


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg) -> None:
    run_experiment(cfg)


if __name__ == "__main__":
    main()
