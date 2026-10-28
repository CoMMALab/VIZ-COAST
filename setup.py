from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vizcoast",
    version="1.0.0",
    author="Muyang Yan, Miras Mengdibayev, Ardon Floros, Weihang Guo, Lydia E. Kavraki, Zachary Kingston",
    description="VIZ-COAST: using VLM reasoning to constrain task and motion planning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CoMMALab/VIZ-COAST",
    packages=find_packages(include=["coast*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy<2",
        "scipy",
        "spatialdyn",
        "ctrlutils",
        "pybullet",
        "pysymbolic",
        "motion_planners@git+https://github.com/branvu/motion-planners.git@0963e2dd02b96dca5c9e5ec908284059cc39053a",
        "openai",
        "google-genai",
        "imageio",
    ],
    extras_require={
        "dev": ["mypy"],
    },
    license="MIT",
    license_files=("LICENSE",),
)