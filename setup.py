import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setuptools.setup(
    name="atomgpt",
    version="2025.4.30",
    author="Kamal Choudhary",
    author_email="kamal.choudhary@nist.gov",
    description="atomgpt",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "atomgpt_forward_train=atomgpt.forward_models.forward_models:main",
            "atomgpt_forward_predict=atomgpt.forward_models.forward_predict:predict",
            "atomgpt_inverse_train=atomgpt.inverse_models.inverse_models:main",
            "atomgpt_inverse_predict=atomgpt.inverse_models.inverse_predict:predict",
        ]
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/usnistgov/atomgpt",
    packages=setuptools.find_packages(),
    license_file="LICENSE.rst",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
