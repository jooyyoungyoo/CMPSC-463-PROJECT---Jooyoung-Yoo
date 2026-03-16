# CMPSC 463 Project — Jooyoung Yoo

## Project Overview
This project was completed for **CMPSC 463: Design and Analysis of Algorithms**.  
The goal is to analyze a real-world **Water Pump RUL (Remaining Useful Life)** dataset using algorithmic techniques rather than machine learning libraries.

The project focuses on the following algorithmic tasks:
- **Divide-and-Conquer Segmentation** of sensor time series
- **Divide-and-Conquer Clustering** of sensor measurements into 4 groups
- **Maximum Subarray (Kadane’s Algorithm)** to detect intervals of strongest sustained deviation
- **Closest Pair** analysis to compare similarity between projected sensor states

The dataset is used to relate sensor behavior to machine-health categories derived from **RUL quantiles**.

## Dataset
Dataset used: **Water Pump RUL – Predictive Maintenance**  
Source: Kaggle

The analysis uses:
- the first **10,000 rows**
- sensor values
- timestamp information
- **rul** (Remaining Useful Life)

RUL values are transformed into 4 condition categories for comparison and evaluation.

## Repository Structure
```text
CMPSC-463-PROJECT---Jooyoung-Yoo/
data/                                   # dataset files
src/                                    # source code for algorithms and analysis
main.py                                 # main entry point
CMPSC 463 Project - Jooyoung Yoo.pdf    # final report
README.md

## How to Run

1. Make sure Python 3 is installed.
2. Clone this repository:
   ```bash
   git clone https://github.com/jooyyoungyoo/CMPSC-463-PROJECT---Jooyoung-Yoo.git
