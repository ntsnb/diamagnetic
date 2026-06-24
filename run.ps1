param(
  [string]$Config = "configs/base.yaml"
)

python train/run_experiment.py --config $Config
