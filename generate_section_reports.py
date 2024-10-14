import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

membership_stats = ['LevelOrdealYouthFinal',
                    'LevelOrdealYoungAdultFinal',
                    'LevelOrdealAdultFinal',
                    'LevelBrotherhoodYouthFinal',
                    'LevelBrotherhoodYoungAdultFinal',
                    'LevelBrotherhoodAdultFinal',
                    'LevelVigilYouthFinal',
                    'LevelVigilYoungAdultFinal',
                    'LevelVigilAdultFinal']

file_paths = [
    "Gateway_PMP_Data_2019.csv",
    "Gateway_PMP_Data_2020.csv",
    "Gateway_PMP_Data_2021.csv",
    "Gateway_PMP_Data_2022.csv",
    "Gateway_PMP_Data_2023.csv"
]

def clean_up_data(dataframes, council_data):
    for df in dataframes:
        df.columns = df.columns.str.strip()
        #print(df.columns)
    council_data.columns = council_data.columns.str.strip()
    #print(council_data.columns)

def consolidate_data(dataframes, council_data): # dataframe of consolidated values
    # Combine all DataFrames into a single DataFrame using pd.concat()
    combined_df = pd.concat(dataframes, ignore_index=True)
    #combined_df.to_csv('combined.csv')





    # Use the 'ID' column in combined_df to map the 'Group' from lookup_df
    #print(combined_df.columns)
    #print(council_data.columns)
    combined_df['CouncilNumber'] = combined_df['CouncilNumber'].fillna(0).astype(int)
    council_data['CouncilNumber'] = council_data['CouncilNumber'].fillna(0).astype(int)



    # Get a list of all column names
    columns = combined_df.columns.tolist()

    # Iterate through the columns and combine if a column and its version with a space exists
    for col in columns:
        if col.endswith(" "):  # Check if the column has a trailing space
            base_col = col.rstrip()  # Remove the trailing space
            if base_col in combined_df.columns:  # If the base column without the space exists
                # Combine the columns using combine_first
                combined_df[base_col] = combined_df[base_col].combine_first(combined_df[col])
                # Drop the column with the space
                combined_df.drop(col, axis=1, inplace=True)



    years = ['2019', '2020', '2021', '2022', '2023']

    # First, handle the 2019 columns since there's no previous year to combine with
    for column in columns:
        if '2019' in column:
            new_name = column.replace('2019', '')
            if new_name not in columns:
                combined_df[new_name] = combined_df[column]  # Just assign 2019 data to the new column
            combined_df.drop(column, axis=1, inplace=True)  # Drop the original 2019 column

    for i in range(1, len(years)):  # Start at year 2020 and go onwards
        for column in combined_df.columns:
            if years[i] in column:
                new_name = column.replace(years[i], '')  # Remove the year from the column name
                combined_df[new_name] = combined_df[new_name].combine_first(combined_df[column])
                
                combined_df.drop(column, axis=1, inplace=True)  # Drop the year-specific column
        
                



    #print(combined_df.columns)   
    #print(council_data.columns)


    # Sort the combined DataFrame by a specific column (e.g., "column_name")
    sorted_df = combined_df.sort_values(by="CouncilNumber")

    mapping_section = council_data.set_index('CouncilNumber')['Section'].to_dict()
    sorted_df['Section'] = sorted_df['CouncilNumber'].map(mapping_section)
    
    #print(council_data.columns)
    mapping_LodgeName = council_data.set_index('CouncilNumber')['LodgeName'].to_dict()
    sorted_df['LodgeName'] = sorted_df['CouncilNumber'].map(mapping_LodgeName)
    
    
    #combined_df.to_csv('raw_consolidated_data_all_sections_all_years.csv')
    # Creating new columns for calculated values
    
    sorted_df['TotalMembers'] = sorted_df[membership_stats].sum(axis=1)
    sorted_df['ElectionRate'] = (sorted_df['Item1ElectionsConducted'] + sorted_df['Item1ElectionsNoEligibleScouts']) / sorted_df['UnitCountFinal'] 
    sorted_df['InductedOrdealTotal'] = sorted_df[['InductedOrdealYouthFinal' ,'InductedOrdealAdultFinal']].sum(axis=1)
    sorted_df['ElectedTotal'] = sorted_df[['ElectedCurrentYouthFinal' ,'ElectedCurrentAdultFinal']].sum(axis=1)
    sorted_df['InductionRate'] = sorted_df['InductedOrdealTotal'] / sorted_df['ElectedTotal']
    sorted_df['ActivationRate'] = sorted_df['Item3ActivatedYouth'] / sorted_df['InductedOrdealTotal'] 
    sorted_df['LevelOrdealTotal'] = sorted_df[['LevelOrdealYouthFinal' ,'LevelOrdealYoungAdultFinal' ,'LevelOrdealAdultFinal']].sum(axis=1)
    sorted_df['LevelBrotherhoodTotal'] = sorted_df[['LevelBrotherhoodYouthFinal' ,'LevelBrotherhoodYoungAdultFinal' ,'LevelBrotherhoodAdultFinal']].sum(axis=1)
    sorted_df['LevelVigilTotal'] = sorted_df[['LevelVigilYouthFinal' ,'LevelVigilYoungAdultFinal' ,'LevelVigilAdultFinal']].sum(axis=1)
    sorted_df['LevelYouthTotal'] = sorted_df[['LevelOrdealYouthFinal' ,'LevelBrotherhoodYouthFinal' ,'LevelVigilYouthFinal']].sum(axis=1)
    sorted_df['LevelYoungAdultTotal'] = sorted_df[['LevelOrdealYoungAdultFinal' ,'LevelBrotherhoodYoungAdultFinal' ,'LevelVigilYoungAdultFinal']].sum(axis=1)
    sorted_df['LevelAdultTotal'] = sorted_df[['LevelOrdealAdultFinal' ,'LevelBrotherhoodAdultFinal' ,'LevelVigilAdultFinal']].sum(axis=1)
    
    
    
    
    
    
    return sorted_df

def isolate_sections(sorted_df):
    
    sections = sorted_df['Section'].unique()  # Get unique sections
    # Define the folder path
    folder_path = "section_reports\\raw_data"

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for section in sections:
        # Filter the DataFrame for rows that belong to the current section
        section_df = sorted_df[sorted_df['Section'] == section]
        
        # Save the filtered DataFrame to a CSV file named by the section
        section_df.to_csv(f'{folder_path}/section_g{section}.csv', index=False)

    print("Data exported by sections into separate CSV files.")
    return True

def iterate_section_reports():
        
    # Specify the folder path
    folder_path = "section_reports\\raw_data"

    # List all files in the folder
    file_list = os.listdir(folder_path)
    file_list = [file for file in file_list if os.path.isfile(os.path.join(folder_path, file))]
    
    for file in file_list:
        generate_section_report(f'{folder_path}\{file}')

def generate_section_report(file_path):
    section_data = pd.read_csv(file_path)
    folder_path = "section_reports\lodge_reports"
    
    
    for lodge in section_data['LodgeName'].unique():  
        with PdfPages(f'{folder_path}\{str(lodge).strip()}_Visual_Report.pdf') as pdf:
            lodge_data_df = section_data[section_data['LodgeName'] == lodge]
            generate_lodge_report(str(lodge).strip(), lodge_data_df, folder_path, pdf)
    
def generate_lodge_report(LodgeName, lodge_data_df, folder_path, pdf):
    #lodge_data_df.set_index('Year', inplace=True)
    lodge_data_df.sort_values(by='Year', inplace=True)
    #lodge_data_df[['ElectionRate', 'InductionRate', 'ActivationRate']].plot(kind='line')
    #plt.show()
        
  
    # Create a figure and a 2x2 grid of subplots, plus one large plot at the bottom
    fig, axs = plt.subplots(3, 2, figsize=(11, 8.5))  # 3 rows, 2 columns
    fig.suptitle(f'{LodgeName}\'s Performance Measurement Program Visual Report', fontsize=16)

    # Top 4 plots (2x2 grid)
    #print(lodge_data_df.columns)
    #print(lodge_data_df['Year'])
    lodge_data_df.plot(x='Year', y=['ElectionRate', 'InductionRate', 'ActivationRate'], ax=axs[0, 0], title='Membership Statistics')
    lodge_data_df.plot(x='Year', y=['PmpOverallPoints'], ax=axs[0, 1], title='Performance Statistics')
    lodge_data_df.plot(x='Year', y=['LevelOrdealTotal','LevelBrotherhoodTotal','LevelVigilTotal'], ax=axs[1, 0], title='Membership by Honor')
    lodge_data_df.plot(x='Year', y=['LevelYouthTotal','LevelYoungAdultTotal','LevelAdultTotal'], ax=axs[1, 1], title='Membership by Age')

    # Bottom plot (one large plot spanning two columns)
    lodge_data_df.plot(x='Year', y=['TotalMembers'], ax=axs[2, :][0], title='Combined Plot')

    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Show plot
    #plt.show()

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)  # Close the figure to avoid displaying it in interactive environments
    



# Load each CSV into a DataFrame and store in a list
dataframes = [pd.read_csv(file) for file in file_paths]

# Alternatively, store in a dictionary for easier reference by year
dataframes_by_year = {year: pd.read_csv(file) for year, file in zip(range(2019, 2024), file_paths)}
council_data = pd.read_csv("Gateway_PMP_Data_Councils.csv")

clean_up_data(dataframes, council_data)

sorted_df = consolidate_data(dataframes, council_data)
isolate_sections(sorted_df)
sorted_df.to_csv('sorted.csv')
    
iterate_section_reports()
    
