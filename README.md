# Quantifying the Time-Varying Contribution of Household Visitors to COVID-19 infections

This repository contains the code for the manuscript:

> **Quantifying the Time-Varying Contribution of Household Visitors to the Spread of COVID-19**  
> Dani Chambon and Martijn H. H. Schoot Uiterkamp

## Data

The analysis uses data from the **COVID RADAR App Dataset**. This dataset is not included in this repository and must be downloaded separately by the user.

The dataset is available through DANS Data Station Life Sciences:

van Dijk, W. J. (2022, October). *COVID RADAR App Dataset*. DANS Data Station Life Sciences. https://doi.org/10.17026/dans-zcd-m9dh

## Reproducing the analysis

1. Clone this repository.
2. Download the COVID RADAR App Dataset from DANS.
3. Place the downloaded data in the appropriate data directory. All scripts in this repository expect the COVID Radar dataset to be found at: "data/COVID_radar_data/CRdataShared.csv"
4. Install the required dependencies.
5. Run the notebooks in the intended order.

## Repository contents

The repository contains the source code used to process the data, perform the statistical analyses, and generate all results presented in the paper.

Reusable model calculations and plotting helpers are kept in `src/model.py` and `src/plotting.py`. Run the notebooks from the repository root so these modules can be imported directly.

## Citation

If you use this repository or the accompanying analyses, please cite the paper, if possible, as well as the COVID RADAR App Dataset.

## Contact

For questions about the analysis or repository, please contact the authors.
