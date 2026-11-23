.PHONY: install test notebooks data lint

install:  ## the package, plus what the notebooks and the tests need
	pip install -e ".[notebooks,dev]"

test:
	pytest -q

notebooks:  ## run every notebook again from a clean kernel, in place (about 20 minutes; notebook 10 needs PyTorch)
	for nb in notebooks/*.ipynb; do \
		jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 $$nb || exit 1; \
	done

data:  ## read the data back out of the figures of the 2014 report
	python tools/digitize_report.py report/figures src/mdplab/data

lint:
	ruff check src tests tools
