import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.ticker as mtick


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
    
    
    
    
    
    sorted_df.sort_values(by='Year', inplace=True)
    return sorted_df

def isolate_sections(sorted_df):
    
    sections = sorted_df['Section'].unique()  # Get unique sections
    # Define the folder path
    folder_path = "all_reports\\section_data"

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

def iterate_sections_for_lodge_reports():
    # Specify the folder path
    folder_path = "all_reports\\section_data"

    # List all files in the folder
    file_list = os.listdir(folder_path)
    file_list = [file for file in file_list if os.path.isfile(os.path.join(folder_path, file))]
    
    for file in file_list:
        generate_lodge_reports_per_section(f'{folder_path}\\{file}')

def generate_lodge_reports_per_section(file_path):
    section_data = pd.read_csv(file_path)

    for lodge in section_data['LodgeName'].unique():
        lodge_data_df = section_data[section_data['LodgeName'] == lodge]
        
        # Get the section name from the lodge data
        lodges_region_str = str(lodge_data_df['Region'][lodge_data_df.index[0]]).strip()
        lodges_section_str = str(lodge_data_df['Section'][lodge_data_df.index[0]]).strip()
        
        # Create a new folder for the section, if it doesn't already exist
        section_folder = f'all_reports\\lodge_reports\\Section_{lodges_region_str}{lodges_section_str}'
        if not os.path.exists(section_folder):
            os.makedirs(section_folder)
        
        # Generate the PDF report for each lodge within the section's folder
        with PdfPages(f'{section_folder}\\{str(lodge).strip()}_Visual_Report.pdf') as pdf:
            generate_lodge_report(str(lodge).strip().replace(" ","_"), lodge_data_df, section_folder, pdf)

def generate_lodge_report(LodgeName, lodge_data_df, folder_path, pdf):
    #lodge_data_df.set_index('Year', inplace=True)
    
    #lodge_data_df[['ElectionRate', 'InductionRate', 'ActivationRate']].plot(kind='line')
    #plt.show()
    
    lodges_region_str = str(lodge_data_df['Region'][lodge_data_df.index[0]]).strip()
    lodges_section_str = lodge_data_df['Section'][lodge_data_df.index[0]]
    
    # Create a figure and a 2x2 grid of subplots, plus one large plot at the bottom
    fig, axs = plt.subplots(3, 2, figsize=(11, 8.5))  # 3 rows, 2 columns
    fig.suptitle(f'{lodges_region_str}{lodges_section_str} - {LodgeName}\'s PMP Visual Report', fontsize=16)

    # Top 4 plots (2x2 grid)
    #print(lodge_data_df.columns)
    #print(lodge_data_df['Year'])
    lodge_data_df.plot(x='Year', y=['ElectionRate', 'InductionRate', 'ActivationRate'], ax=axs[0, 0])
    lodge_data_df.plot(x='Year', y=['PmpOverallPoints'], ax=axs[0, 1])
    lodge_data_df.plot(x='Year', y=['LevelOrdealTotal','LevelBrotherhoodTotal','LevelVigilTotal'], kind='bar', stacked=True, ax=axs[1, 0])
    lodge_data_df.plot(x='Year', y=['LevelYouthTotal','LevelYoungAdultTotal','LevelAdultTotal'], kind='bar', stacked=True, ax=axs[1, 1])
    
    lodge_data_df.plot(x='Year', y=['TotalMembers'], ax=axs[2, :][0], title='Total Membership')

    axs[0,0].set_ylabel('Rate (%)')
    axs[0,1].set_ylabel('Points')
    axs[1,0].set_ylabel('Members')
    axs[1,1].set_ylabel('Members')
    
    axs[0,0].set_title('Membership Statistics')
    axs[0,1].set_title('PMP Points')
    axs[1,0].set_title('Membership by Honor')
    axs[1,1].set_title('Membership by Age')
    
    axs[0,0].xaxis.set_major_locator(mtick.MaxNLocator(nbins=5))  # Set the maximum number of ticks to 5
    axs[0,1].xaxis.set_major_locator(mtick.MaxNLocator(nbins=5))  # Set the maximum number of ticks to 5
    axs[1,0].xaxis.set_major_locator(mtick.MaxNLocator(nbins=5))  # Set the maximum number of ticks to 5
    axs[1,1].xaxis.set_major_locator(mtick.MaxNLocator(nbins=5))  # Set the maximum number of ticks to 5
    
    axs[0,0].legend(loc='upper left', bbox_to_anchor=(1,1))
    axs[0,1].legend(loc='upper left', bbox_to_anchor=(1,1))
    axs[1,0].legend(loc='upper left', bbox_to_anchor=(1,1))
    axs[1,1].legend(loc='upper left', bbox_to_anchor=(1,1))
    
    axs[0,0].set_ylim(0, None)
    axs[0,1].set_ylim(0, None)
    #axs[1,0].set_ylim(0, None)
    #axs[1,1].set_ylim(0, None)
    axs[2,0].set_ylim(0, None)
    
    axs[0, 0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))  # xmax=1 for percentage from 0 to 100
    

    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 1])

    #plt.show()

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)  # Close the figure to avoid displaying it in interactive environments

"""
def get_five_num_summary(arr):
    
    min_val = np.min(arr)
    one_qr_val = np.percentile(arr, 25)
    med_val = np.median(arr)
    thr_qr_val = np.percentile(arr, 75)
    max_val = np.max(arr)
    
    return (min_val, one_qr_val, med_val, thr_qr_val, max_val)
"""

def all_reports(sorted_df):
    
    """sections = sorted_df.groupby(['Section'])[
    ['Item8CapitalContributionsCash', 'PmpOverallPoints', 'Item8CapitalContributionsMaterial' ,'Item8EndowmentContributions', 
     'Item8FosContributions', 'LevelOrdealYouthFinal', 'LevelOrdealYoungAdultFinal', 'LevelOrdealAdultFinal', 
     'LevelBrotherhoodYouthFinal', 'LevelBrotherhoodYoungAdultFinal', 'LevelBrotherhoodAdultFinal', 'LevelVigilYouthFinal', 
     'LevelVigilYoungAdultFinal', 'LevelVigilAdultFinal', 'InductedOrdealYouthFinal', 'InductedOrdealAdultFinal', 
     'InductedBrotherhoodYouthFinal', 'InductedBrotherhoodAdultFinal', 'InductedVigilYouthFinal', 'InductedVigilAdultFinal', 
     'ElectedCurrentYouthFinal', 'ElectedCurrentAdultFinal', 'ElectedPreviousYouthFinal', 'ElectedPreviousAdultFinal', 
     'UnitCountFinal', 'Item1ElectionsConducted', 'Item1ElectionsNoEligibleScouts', 'Item1VisitsConducted', 'Item3ActivatedYouth', 
     'Item9ServiceHours', 'TotalMembers', 'InductedOrdealTotal', 'ElectedTotal', 'LevelOrdealTotal', 'LevelBrotherhoodTotal', 
     'LevelVigilTotal', 'LevelYouthTotal', 'LevelYoungAdultTotal', 'LevelAdultTotal']
    ].sum().reset_index()"""
    
    sections = sorted_df.groupby(['Year', 'Section']).sum().reset_index()
    # Assuming 'sections' is your DataFrame
    columns_to_delete = ['CharterApplicationId', 'Item8EndowmentContributions', 'Item8FosContributions', 'CouncilNumber', 'SequenceNumber', 'Timestamp', 'DateSubmitted', 'PmpOverallLevel', 'LodgeName', 'Item8CapitalContributionsCash', 'Item8CapitalContributionsMaterial']
    sections = sections.drop(columns=columns_to_delete, errors='ignore')

    
    
    
    folder_path = f'all_reports\section_data'
    file_list = os.listdir(folder_path)
    for current_file in file_list:
        
        election_rate_year = []
        induction_rate_year = []
        activation_rate_year = []
        section_df = pd.read_csv(f'{folder_path}\{current_file}')
        for year in section_df['Year'].unique():
            election_rate_vals = section_df[section_df['Year'] == year]['ElectionRate'].to_numpy()
            induction_rate_vals = section_df[section_df['Year'] == year]['InductionRate'].to_numpy()
            activation_rate_vals = section_df[section_df['Year'] == year]['ActivationRate'].to_numpy()
            
            election_rate_year += [election_rate_vals]
            induction_rate_year += [induction_rate_vals]
            activation_rate_year += [activation_rate_vals]
                
                
        sections['TotalMembers'] = sections[membership_stats].sum(axis=1)
        sections['ElectionRate'] = (sections['Item1ElectionsConducted'] + sections['Item1ElectionsNoEligibleScouts']) / sections['UnitCountFinal'] 
        sections['InductedOrdealTotal'] = sections[['InductedOrdealYouthFinal' ,'InductedOrdealAdultFinal']].sum(axis=1)
        sections['ElectedTotal'] = sections[['ElectedCurrentYouthFinal' ,'ElectedCurrentAdultFinal']].sum(axis=1)
        sections['InductionRate'] = sections['InductedOrdealTotal'] / sections['ElectedTotal']
        sections['ActivationRate'] = sections['Item3ActivatedYouth'] / sections['InductedOrdealTotal'] 
        sections['LevelOrdealTotal'] = sections[['LevelOrdealYouthFinal' ,'LevelOrdealYoungAdultFinal' ,'LevelOrdealAdultFinal']].sum(axis=1)
        sections['LevelBrotherhoodTotal'] = sections[['LevelBrotherhoodYouthFinal' ,'LevelBrotherhoodYoungAdultFinal' ,'LevelBrotherhoodAdultFinal']].sum(axis=1)
        sections['LevelVigilTotal'] = sections[['LevelVigilYouthFinal' ,'LevelVigilYoungAdultFinal' ,'LevelVigilAdultFinal']].sum(axis=1)
        sections['LevelYouthTotal'] = sections[['LevelOrdealYouthFinal' ,'LevelBrotherhoodYouthFinal' ,'LevelVigilYouthFinal']].sum(axis=1)
        sections['LevelYoungAdultTotal'] = sections[['LevelOrdealYoungAdultFinal' ,'LevelBrotherhoodYoungAdultFinal' ,'LevelVigilYoungAdultFinal']].sum(axis=1)
        sections['LevelAdultTotal'] = sections[['LevelOrdealAdultFinal' ,'LevelBrotherhoodAdultFinal' ,'LevelVigilAdultFinal']].sum(axis=1)
        
        
        #sections.plot('Year', )
        
        lodges_region_str = str(sections['Region'][sections.index[0]]).strip()
        lodges_section_str = sections['Section'][sections.index[0]]
        
        for section in sections['Section'].unique():
            with PdfPages(f'all_reports\section_reports\G{str(section).strip()}_Visual_Report.pdf') as pdf:
                section_df = sections[sections['Section'] == section]
                # Create a figure and a 2x2 grid of subplots, plus one large plot at the bottom
                fig, axs = plt.subplots(3, 2, figsize=(11, 8.5))  # 3 rows, 2 columns
                fig.suptitle(f'{lodges_region_str[0]}{section}\'s Section PMP Visual Report', fontsize=16)

            
                section_df.plot(x='Year', y=['ElectionRate'], ax=axs[0, 0])
                section_df.plot(x='Year', y=['InductionRate'], ax=axs[1, 0])
                section_df.plot(x='Year', y=['ActivationRate'], ax=axs[2, 0])
                
                ax00 = axs[0,0].twinx()
                ax10 = axs[1,0].twinx()
                ax20 = axs[2,0].twinx()
                
                boxplot_positions = section_df['Year'].unique()
                print(boxplot_positions)
                #print(f'Election Rate {election_rate_year}')
                for i in election_rate_year:
                    print(i)
                    print('new row')
                print(len(election_rate_year))  # Check number of rows in election_rate_year
                print(len(boxplot_positions))   # Check number of positions

                #print(f'Induction Rate {np.shape(induction_rate_year)}')
                #print(f'Activation Rate {np.shape(activation_rate_year)}')
                
                ax00.boxplot(election_rate_year, positions=boxplot_positions, widths=0.4)
                ax10.boxplot(induction_rate_year, positions=boxplot_positions, widths=0.4)
                ax20.boxplot(activation_rate_year, positions=boxplot_positions, widths=0.4)
                
                section_df.plot(x='Year', y=['PmpOverallPoints'], ax=axs[0, 1])
                section_df.plot(x='Year', y=['LevelOrdealTotal','LevelBrotherhoodTotal','LevelVigilTotal'], kind='bar', stacked=True, ax=axs[1, 1])
                section_df.plot(x='Year', y=['LevelYouthTotal','LevelYoungAdultTotal','LevelAdultTotal'], kind='bar', stacked=True, ax=axs[2, 1])
                
                axs[0,0].set_ylabel('Rate (%)')
                axs[1,0].set_ylabel('Rate (%)')
                axs[2,0].set_ylabel('Rate (%)')
                axs[0,1].set_ylabel('Members')
                axs[1,1].set_ylabel('Members')
                
                axs[0,0].set_title('Election Rate')
                axs[1,0].set_title('Induction Rate')
                axs[2,0].set_title('Activation Rate')
                axs[1,1].set_title('Membership by Honor')
                axs[2,1].set_title('Membership by Age')
                
                axs[0,0].xaxis.set_major_locator(mtick.MaxNLocator(nbins=5))  # Set the maximum number of ticks to 5
                axs[0,1].xaxis.set_major_locator(mtick.MaxNLocator(nbins=5))  # Set the maximum number of ticks to 5
                axs[1,0].xaxis.set_major_locator(mtick.MaxNLocator(nbins=5))  # Set the maximum number of ticks to 5
                axs[1,1].xaxis.set_major_locator(mtick.MaxNLocator(nbins=5))  # Set the maximum number of ticks to 5
                
                axs[0,0].legend(loc='upper left', bbox_to_anchor=(1,1))
                axs[0,1].legend(loc='upper left', bbox_to_anchor=(1,1))
                axs[1,0].legend(loc='upper left', bbox_to_anchor=(1,1))
                axs[1,1].legend(loc='upper left', bbox_to_anchor=(1,1))
                
                axs[0,0].set_ylim(0, None)
                axs[0,1].set_ylim(0, None)
                axs[1,0].set_ylim(0, None)
                #axs[1,1].set_ylim(0, None)
                axs[2,0].set_ylim(0, None)
                
                axs[0, 0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))  # xmax=1 for percentage from 0 to 100
                axs[1, 0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))  # xmax=1 for percentage from 0 to 100
                axs[2, 0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))  # xmax=1 for percentage from 0 to 100
                #sections.to_csv(f'section_G{section}.csv')
                plt.show()
                pdf.savefig(fig)
            

# Load each CSV into a DataFrame and store in a list
dataframes = [pd.read_csv(file) for file in file_paths]

# Alternatively, store in a dictionary for easier reference by year
dataframes_by_year = {year: pd.read_csv(file) for year, file in zip(range(2019, 2024), file_paths)}
council_data = pd.read_csv("Gateway_PMP_Data_Councils.csv")

clean_up_data(dataframes, council_data)

sorted_df = consolidate_data(dataframes, council_data)

isolate_sections(sorted_df)

sorted_df.to_csv('sorted.csv')
    
#all_reports(sorted_df)
iterate_sections_for_lodge_reports()
    





# - - - - For shits and giggles - - - - -
# count the number of scouts across the 
#print(sorted_df.groupby('Year')['TotalMembers'].sum().reset_index())

