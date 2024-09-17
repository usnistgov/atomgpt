import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="atomgpt",
    version="2024.9.8",
    author="Kamal Choudhary",
    author_email="kamal.choudhary@nist.gov",
    description="atomgpt",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.6.3",
        "jarvis-tools>=2021.07.19",
        "transformers",
        "pydantic_settings",
        "peft",
        "trl",
        "triton",
        "torch",
        "sentencepiece",
        "protobuf",
        # "alignn",
    ],
    # scripts=["atomgpt/train_prop.py"],
    entry_points={
        "console_scripts": [
            "atomgpt_forward=atomgpt.forward_models.forward_models:main",
            "atomgpt_inverse=atomgpt.inverse_models.inverse_models:main",
        ]
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/usnistgov/atomgpt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
