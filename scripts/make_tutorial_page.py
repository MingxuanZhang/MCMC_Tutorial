import nbsphinx
rst, res = nbsphinx.Exporter().from_filename('mcmc_tutorial.ipynb')
rst = rst.split("\n")
fileID = open("docs/source/tutorial.rst", "w")
for line in rst:
	fileID.write(line + "\n")
fileID.close()