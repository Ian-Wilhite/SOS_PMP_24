import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.ticker as mtick
from PyPDF2 import PdfMerger, PdfReader
from PyPDF2.errors import PdfReadError, EmptyFileError



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
    "input_files\Gateway_PMP_Data_2019.csv",
    "input_files\Gateway_PMP_Data_2020.csv",
    "input_files\Gateway_PMP_Data_2021.csv",
    "input_files\Gateway_PMP_Data_2022.csv",
    "input_files\Gateway_PMP_Data_2023.csv"
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
    sorted_df['ElectedTotal'] = sorted_df[['ElectedCurrentYouthFinal' ,'ElectedCurrentAdultFinal', 'ElectedPreviousYouthFinal', 'ElectedPreviousAdultFinal']].sum(axis=1)
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

def section_reports(sorted_df):
    # Group by section and year, removing unnecessary columns
    sections = sorted_df.groupby(['Year', 'Section']).sum().reset_index()  # Ensure Year remains after grouping
    
    columns_to_delete = ['CharterApplicationId', 'Item8EndowmentContributions', 'Item8FosContributions', 
                         'CouncilNumber', 'SequenceNumber', 'Timestamp', 'DateSubmitted', 
                         'PmpOverallLevel', 'LodgeName', 'Item8CapitalContributionsCash', 
                         'Item8CapitalContributionsMaterial']
    sections = sections.drop(columns=columns_to_delete, errors='ignore')

    # Compute additional columns
    sections['TotalMembers'] = sections[['InductedOrdealYouthFinal', 'InductedOrdealAdultFinal']].sum(axis=1)
    sections['ElectionRate'] = (sections['Item1ElectionsConducted'] + sections['Item1ElectionsNoEligibleScouts']) / sections['UnitCountFinal']
    sections['InductionRate'] = sections['InductedOrdealTotal'] / sections['ElectedTotal']
    sections['ActivationRate'] = sections['Item3ActivatedYouth'] / sections['InductedOrdealTotal']
    
    # Fill NaN values that may arise during computation to avoid invalid operations
    sections['ElectionRate'].fillna(0, inplace=True)
    sections['InductionRate'].fillna(0, inplace=True)
    sections['ActivationRate'].fillna(0, inplace=True)

    # Summing levels
    sections['LevelOrdealTotal'] = sections[['LevelOrdealYouthFinal', 'LevelOrdealYoungAdultFinal', 'LevelOrdealAdultFinal']].sum(axis=1)
    sections['LevelBrotherhoodTotal'] = sections[['LevelBrotherhoodYouthFinal', 'LevelBrotherhoodYoungAdultFinal', 'LevelBrotherhoodAdultFinal']].sum(axis=1)
    sections['LevelVigilTotal'] = sections[['LevelVigilYouthFinal', 'LevelVigilYoungAdultFinal', 'LevelVigilAdultFinal']].sum(axis=1)
    sections['LevelYouthTotal'] = sections[['LevelOrdealYouthFinal', 'LevelBrotherhoodYouthFinal', 'LevelVigilYouthFinal']].sum(axis=1)
    sections['LevelYoungAdultTotal'] = sections[['LevelOrdealYoungAdultFinal', 'LevelBrotherhoodYoungAdultFinal', 'LevelVigilYoungAdultFinal']].sum(axis=1)
    sections['LevelAdultTotal'] = sections[['LevelOrdealAdultFinal', 'LevelBrotherhoodAdultFinal', 'LevelVigilAdultFinal']].sum(axis=1)

    # Iterate over section files
    folder_path = f'all_reports/section_data'
    file_list = os.listdir(folder_path)
    
    for current_file in file_list:
        section_name = current_file[:-4].split('_')[1][1:]
        # Filter the data for the specific section
        section_data = sorted_df[sorted_df['Section'] == section_name]
        
        if section_data.empty:
            print(f"Section {section_name} has no data. Skipping.")
            continue  # Skip to the next file if no data for the section

        cap_threshold = 1.5  # 150%

        section_data.loc[:, 'ElectionRate'] = section_data['ElectionRate'].apply(lambda x: min(x, cap_threshold))
        section_data.loc[:, 'InductionRate'] = section_data['InductionRate'].apply(lambda x: min(x, cap_threshold))
        section_data.loc[:, 'ActivationRate'] = section_data['ActivationRate'].apply(lambda x: min(x, cap_threshold))
                
        # Gather data for plotting
        election_rate_year = []
        induction_rate_year = []
        activation_rate_year = []
        
        for year in section_data['Year'].unique():
            election_rate_vals = section_data[section_data['Year'] == year]['ElectionRate'].to_numpy()
            induction_rate_vals = section_data[section_data['Year'] == year]['InductionRate'].to_numpy()
            activation_rate_vals = section_data[section_data['Year'] == year]['ActivationRate'].to_numpy()

            election_rate_year.append(election_rate_vals)
            induction_rate_year.append(induction_rate_vals)
            activation_rate_year.append(activation_rate_vals)

        # For weighted average lines
        weighted_avg_election_rate = sections.groupby('Year').apply(lambda x: np.average(x['ElectionRate'], weights=x['TotalMembers']))
        weighted_avg_induction_rate = sections.groupby('Year').apply(lambda x: np.average(x['InductionRate'], weights=x['TotalMembers']))
        weighted_avg_activation_rate = sections.groupby('Year').apply(lambda x: np.average(x['ActivationRate'], weights=x['TotalMembers']))

        # Create plots
        fig, axs = plt.subplots(3, 2, figsize=(11, 8.5))  # 3 rows, 2 columns
        fig.suptitle(f'G{section_name}\'s Section PMP Visual Report', fontsize=16)
        
        boxplot_positions = section_data['Year'].unique()

        # Election Rate Plot
        axs[0, 0].boxplot(election_rate_year, positions=boxplot_positions, widths=0.4)
        axs[0, 0].plot(boxplot_positions, weighted_avg_election_rate, color='blue', label='Weighted Avg Election Rate')
        axs[0, 0].set_title('Election Rate')
        axs[0, 0].set_ylabel('Rate (%)')
        axs[0, 0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

        # Induction Rate Plot
        axs[1, 0].boxplot(induction_rate_year, positions=boxplot_positions, widths=0.4)
        axs[1, 0].plot(boxplot_positions, weighted_avg_induction_rate, color='blue', label='Weighted Avg Induction Rate')
        axs[1, 0].set_title('Induction Rate')
        axs[1, 0].set_ylabel('Rate (%)')
        axs[1, 0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

        # Activation Rate Plot
        axs[2, 0].boxplot(activation_rate_year, positions=boxplot_positions, widths=0.4)
        axs[2, 0].plot(boxplot_positions, weighted_avg_activation_rate, color='blue', label='Weighted Avg Activation Rate')
        axs[2, 0].set_title('Activation Rate')
        axs[2, 0].set_ylabel('Rate (%)')
        axs[2, 0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

        # Additional plots
        section_by_year = section_data.groupby('Year').sum().reset_index()  # Ensure Year is present
        section_by_year.plot(x='Year', y=['PmpOverallPoints'], ax=axs[0, 1])
        section_by_year.plot(x='Year', y=['LevelOrdealTotal', 'LevelBrotherhoodTotal', 'LevelVigilTotal'], kind='bar', stacked=True, ax=axs[1, 1])
        section_by_year.plot(x='Year', y=['LevelYouthTotal', 'LevelYoungAdultTotal', 'LevelAdultTotal'], kind='bar', stacked=True, ax=axs[2, 1])

        # Set labels and titles
        axs[0, 1].set_ylabel('Points')
        axs[1, 1].set_ylabel('Members')
        axs[2, 1].set_ylabel('Members')
        axs[1, 1].set_title('Membership by Honor')
        axs[2, 1].set_title('Membership by Age')

        # Adjust layout and close figure after saving
        plt.tight_layout()
        with PdfPages(f'all_reports/section_reports/G{str(section_name).strip()}_Visual_Report.pdf') as pdf:
            pdf.savefig(fig)
            plt.close(fig)  # Close figure to avoid memory issues

def combine_lodge_reports_into_pdf(root_folder, output_pdf):
    # Create a PdfMerger object
    pdf_merger = PdfMerger()

    # Iterate through section folders inside the "all_reports" folder
    for section_folder in sorted(os.listdir(root_folder)):
        section_path = os.path.join(root_folder, section_folder)
        
        # Check if it's a folder (to ignore any files)
        if os.path.isdir(section_path):
            # Iterate through lodge report PDFs in each section folder
            for lodge_report in sorted(os.listdir(section_path)):
                if lodge_report.endswith('.pdf'):
                    lodge_report_path = os.path.join(section_path, lodge_report)
                    print(f'Adding {lodge_report_path} to the combined PDF.')
                    # Append each lodge report to the merger
                    with open(lodge_report_path, 'rb') as f:
                        pdf_merger.append(f)

    # Write the combined PDF to the output file
    with open(output_pdf, 'wb') as output_file:
        pdf_merger.write(output_file)

    print(f'Combined PDF saved as {output_pdf}')

def combine_section_reports_into_pdf(section_root_folder, section_output_pdf):
    # Initialize a PdfMerger object
    pdf_merger = PdfMerger()

    # Get all PDF files in the section reports folder
    pdf_files = [file_name for file_name in os.listdir(section_root_folder) if file_name.endswith('.pdf')]

    # Sort by numeric part of the file name (e.g., 'g1', 'g2', etc.)
    def extract_section_number(file_name):
        match = re.search(r'g(\d+)', file_name)
        return int(match.group(1)) if match else float('inf')  # Default to infinity if no match

    pdf_files_sorted = sorted(pdf_files, key=extract_section_number)

    # Loop through sorted PDF files
    for file_name in pdf_files_sorted:
        file_path = os.path.join(section_root_folder, file_name)

        # Check if the file is not empty
        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty file: {file_path}")
            continue

        # Try to append the PDF, and handle any errors (like corrupt PDFs)
        try:
            with open(file_path, 'rb') as pdf_file:
                reader = PdfReader(pdf_file)
                if len(reader.pages) == 0:
                    print(f"Skipping empty PDF file: {file_path}")
                    continue
                pdf_merger.append(pdf_file)
        except (PdfReadError, EmptyFileError) as e:
            print(f"Skipping corrupt or unreadable file: {file_path}, Error: {e}")
            continue
    
    # Write the combined PDF to the specified output path
    with open(section_output_pdf, 'wb') as output_pdf:
        pdf_merger.write(output_pdf)

    print(f"Combined PDF saved as: {section_output_pdf}")
 
# Load each CSV into a DataFrame and store in a list
dataframes = [pd.read_csv(file) for file in file_paths]

# Alternatively, store in a dictionary for easier reference by year
dataframes_by_year = {year: pd.read_csv(file) for year, file in zip(range(2019, 2024), file_paths)}
council_data = pd.read_csv("input_files\Gateway_PMP_Data_Councils.csv")

clean_up_data(dataframes, council_data)

sorted_df = consolidate_data(dataframes, council_data)

# Check for NaN and Infinite values
#print(sorted_df.isnull().sum())  # Check for NaN values

sorted_df = sorted_df.dropna()  # Drop rows with any NaN values

numeric_df = sorted_df.select_dtypes(include=[np.number])


plt.matshow(numeric_df.corr())
continuous_features = numeric_df.columns
#plt.xticks(range(len(continuous_features)), continuous_features, rotation="45")
#plt.yticks(range(len(continuous_features)), continuous_features, rotation="45")
plt.colorbar()
plt.show()

for i, col in enumerate(numeric_df.columns):
    print(f'{i}: {col}')
# sorted_df.describe()

# plt.matshow(sorted_df.corr())
# continuous_features = sorted_df.describe().columns
# plt.xticks(range(len(continuous_features)), continuous_features, rotation="45")
# plt.yticks(range(len(continuous_features)), continuous_features, rotation="45")
# plt.colorbar()
# plt.show()

# isolate_sections(sorted_df)

# sorted_df.to_csv('sorted.csv')
    
# section_reports(sorted_df)

# #Generate all lodge reports
# #iterate_sections_for_lodge_reports()

# #create massive PDF for printing
# lodge_root_folder = 'all_reports/lodge_reports'
# lodge_output_pdf = 'all_reports/combined_lodge_reports.pdf'
# #combine_reports_into_pdf(root_folder, output_pdf)

# section_root_folder = 'all_reports/section_reports'
# section_output_pdf = 'all_reports/combined_section_reports.pdf'
# combine_section_reports_into_pdf(section_root_folder, section_output_pdf)

# - - - - For shits and giggles - - - - -
# count the number of scouts across the 
#print(sorted_df.groupby('Year')['TotalMembers'].sum().reset_index())

