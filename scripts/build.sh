# Make tutorial page
rm docs/source/tutorial.rst
touch docs/source/tutorial.rst
python scripts/make_tutorial_page.py
# Build sphinx!
sphinx-build docs/source docs/build/html