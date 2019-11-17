import setuptools

with open("README.md", "r") as readme:
    long_description = readme.read()

setuptools.setup(
    name="espn_fantasy_rugby",
    version="0.0.1",
    author="Niclas Thomas",
    author_email="niclas.thomas@gmail.com",
    description="ESPN Fantasy Rugby Selector",
    long_description=long_description,
    packages=setuptools.find_packages()
)