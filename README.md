# ModelSSCOutflows
Contains scripts and input files used in the modeling presented by [Levy et al. 2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...912....4L/abstract).

*Any use of this code or data must reference [Levy et al. 2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...912....4L/abstract).*

## Getting started
1. Download all files and folders. 
2. Check that the dependent packages are installed: argparse, astropy, datetime, matplotlib, mpl_toolkits, numpy, pandas, os, scipy, sys, time
3. In a python 3 terminal, display the help file: ``>> python ModelOutflowPCygni.py -h``

Example: `>> python ModelOutflowPCygni.py '14' 'CS(7-6)' './'`

## Troubleshooting
1. The code should check all paths and create them where possible. Check for errors and exit messages where paths are broken/missing.
2. Check the list of required packages (`>> python ModelOutflowPCygni.py -h` and/or `>> python ComparePCygniModel.py -h`).
    * Code tested using Python 3.7.4, iPython 7.13.0, argparse 1.1, astropy 3.2.1, matplotlib 3.1.3, numpy 1.16.4, pandas 1.0.3, scipy 1.4.1 
3. Consult [the paper](https://ui.adsabs.harvard.edu/abs/2021ApJ...912....4L/abstract), especially Appendix B.
4. Email the author Rebecca Levy (rlevy.astro@gmail.com).

## Notes
This repository's main branch is called [main, not master](https://www.cnet.com/news/microsofts-github-is-removing-coding-terms-like-master-and-slave/).
