# ModelSSCOutflows
Contains scripts and input files used in the modeling presented by [Levy et al. 2020](https://ui.adsabs.harvard.edu/abs/2020arXiv201105334L/abstract). Any use of this code or data must reference [Levy et al. 2020](https://ui.adsabs.harvard.edu/abs/2020arXiv201105334L/abstract).

## Getting started
1. Download all files and folders. 
2. Check that the dependent packages are installed: argparse, astropy, datetime, matplotlib, mpl_toolkits, numpy, pandas, os, scipy, sys, time
3. In a python 3 terminal: ``>> python ModelOutflowPCygni.py -h``

Example: `>> python ModelOutflowPCygni.py '14' 'CS(7-6)' './'`

## Troubleshooting
- The code should check all paths and create them where possible. Check for errors and exit messages where paths are broken/missing.
- Check the list of required packages (`>> python ModelOutflowPCygni.py -h` and/or `>> python ComparePCygniModel.py -h`).
- Consult [the paper](https://ui.adsabs.harvard.edu/abs/2020arXiv201105334L/abstract).
- Email the author Rebecca Levy (rlevy.astro@gmail.com).

## Notes
This repository's main branch is called [main, not master](https://www.cnet.com/news/microsofts-github-is-removing-coding-terms-like-master-and-slave/).
