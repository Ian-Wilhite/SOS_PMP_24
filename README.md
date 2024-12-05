# Lodge Data Visualization and Report Generation

This project consolidates and visualizes lodge data across the country and generates detailed reports for individual sections and lodges. This README provides guidance for both non-technical users and developers looking to maintain or extend the codebase.

---

## Table of Contents
1. [User Guide: Accessing Files and Reports on GitHub](#user-guide-accessing-files-and-reports-on-github)
   - [Repository Overview](#1-repository-overview)
   - [Steps to Access Reports](#2-steps-to-access-reports)
   - [Generated Report Structure](#3-generated-report-structure)
2. [Developer Guide: How the Code Works](#developer-guide-how-the-code-works)
   - [Data Cleaning and Consolidation](#1-data-cleaning-and-consolidation)
   - [Data Export by Section](#2-data-export-by-section)
   - [Report Generation](#3-report-generation)
   - [Error Handling](#4-error-handling)
   - [Dependencies](#5-dependencies)
   - [Extending the Code](#6-extending-the-code)
3. [Contact](#contact)
4. [Contributions](#contributions)

---

## User Guide: Accessing Files and Reports on GitHub

### 1. Repository Overview
The GitHub repository contains the following key folders and files:
- **`input_files/`**: Raw data files used for processing.
- **`all_reports/`**: Generated reports organized by sections and lodges.
- **`scripts/`**: Python scripts for data processing and report generation.

### 2. Steps to Access Reports
1. Navigate to the **`all_reports/`** folder in the repository.
2. Inside **`all_reports/`**, you'll find:
   - **`section_data/`**: CSV files with data grouped by sections.
   - **`lodge_reports/`**: PDF reports for each lodge, organized into subfolders by section.
3. Download the desired files directly from GitHub.

### 3. Generated Report Structure
Each lodge's report (PDF) includes:
- Membership trends by honor (Ordeal, Brotherhood, Vigil).
- Membership breakdown by age (Youth, Young Adult, Adult).
- Metrics such as Election Rate, Induction Rate, and Activation Rate over time.
- Total membership statistics.

Sample Section Report:
[View the Project Documentation](resources/G3_Visual_Report-1.png)

Sample Lodge Report:
[View the Project Documentation](resources/Netopalis%20Sipo%20Schipinachk_Visual_Report-1.png)


---

## Developer Guide: How the Code Works

This section provides a high-level overview of the codebase for maintenance and extensions.

### 1. **Data Cleaning and Consolidation**
- **Files:** `scripts/consolidate_data.py`
- **Key Steps:**
  - Load multiple input CSV files (`input_files/`) and clean column names.
  - Consolidate data across years by combining year-specific columns into unified metrics (e.g., membership stats).
  - Map additional information (e.g., Section, Lodge Name) using lookup tables.

### 2. **Data Export by Section**
- **Function:** `isolate_sections(sorted_df)`
- **Output:** Exports CSV files for each section into **`all_reports/section_data/`**.

### 3. **Report Generation**
- **Function:** `generate_lodge_reports_per_section(file_path)`
- **Output:** Creates PDF visual reports for each lodge in the **`all_reports/lodge_reports/`** directory.
- **Visualization:** 
  - Plots include election/induction/activation rates, membership by honor and age, and total membership trends.
  - Data is visualized using `matplotlib` and saved as multi-page PDFs.

### 4. **Error Handling**
- Handles common issues such as missing data, empty files, or read errors using `try-except` blocks.

### 5. **Dependencies**
The code requires the following Python libraries:
- `pandas` (data manipulation)
- `matplotlib` (visualization)
- `PyPDF2` (PDF merging and handling)
- `os` and `numpy` (file handling and numerical operations)

### 6. **Extending the Code**
To add new features or adjust existing ones:
- Update **`membership_stats`** or other metric definitions in `consolidate_data()`.
- Modify or add visualization logic in `generate_lodge_report()`.

---

## Contributions
We welcome contributions from the community! Here's how you can help:
1. **Report Bugs:** Open an issue in the repository with details about the problem.
2. **Suggest Features:** Submit ideas for improvements or new features.
3. **Submit Pull Requests:** 
   - Fork the repository and create a feature branch.
   - Make your changes and commit with clear messages.
   - Submit a pull request for review.

## Contact
For questions or support, you can contact me at iwilhite@netopalis.org 

WWW
Ian
