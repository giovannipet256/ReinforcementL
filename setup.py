from setuptools import setup, find_packages

setup(
    name="hospital_scheduler",
    version="2.0.0",
    description="RL-based hospital shift scheduling (refactored)",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=0.29",
        "numpy>=1.24",
        "torch==2.7.1",
        "stable-baselines3>=2.0",
        "sb3-contrib>=2.0",
        "pandas>=2.0",
        "streamlit>=1.30",
    ],
    entry_points={
        "console_scripts": [
            "hs-train=hospital_scheduler.scripts.train:main",
            "hs-run=hospital_scheduler.scripts.run:main",
            "hs-eval=hospital_scheduler.scripts.eval:main",
        ]
    },
)
