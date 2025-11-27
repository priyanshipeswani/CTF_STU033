# CTF_STU033 - Capture the Flag Challenge

## Overview
This repository contains the solution for a 3-step AI detective challenge to identify a manipulated book review in a dataset of books and reviews.

## Challenge Steps
1. **FLAG1**: Found the manipulated book by computing SHA256("STU033") hash and locating it in reviews
2. **FLAG2**: Identified the fake review containing the hash signature  
3. **FLAG3**: Used SHAP analysis on a trained classifier to identify authentic review patterns

## Files
- `flag.py` - FLAG1 and FLAG2 solution
- `flag3_solution.py` - FLAG3 SHAP analysis solution
- `flags.txt` - All three discovered flags
- `reflection.md` - Detailed methodology explanation
- `books.csv` & `reviews.csv` - Dataset files

## Usage
Run the Python scripts to reproduce the flag discovery process.