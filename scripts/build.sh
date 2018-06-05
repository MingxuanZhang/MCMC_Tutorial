# Copy figures over so that notebook page will render them
rm -rf docs/source/figs
mkdir docs/source/figs
cp -r figs/ docs/source/figs
# Build sphinx!
sphinx-build docs/source docs/build/html