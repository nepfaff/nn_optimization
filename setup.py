from distutils.core import setup

setup(
    name="nn_optimization",
    version="0.0.0",
    packages=["nn_optimization"],
    install_requires=["numpy", "hydra-core", "wandb", "omegaconf"],
)
