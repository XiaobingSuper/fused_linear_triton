# Create a project with the name xiaobing_triton_ops and the following structure:


from setuptools import find_packages, setup

print(find_packages())

setup(
    name="xiaobing_triton_ops",
    packages=find_packages(),
    version="0.1.0",
    install_requires=["torch"],
)


