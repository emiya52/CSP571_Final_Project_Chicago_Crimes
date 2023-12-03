# Load necessary libraries
library(dplyr)
library(tidyr)
library(readr)

# Set the path to your combined CSV file
combined_file_path <- "Crimes_-_2001_to_Present.csv"

# Read the entire dataset
df <- read_csv(combined_file_path)

# Define the cleaning function
clean_data <- function(df) {
  df %>%
    select(-ID, -`Case Number`, -IUCR, -Description, -`FBI Code`, -`Updated On`) %>%
    filter(
      !is.na(`Date`) & !is.na(`Location Description`) & 
        !is.na(`Primary Type`) & !is.na(Arrest) & 
        !is.na(Domestic) & !is.na(`Community Area`)
    ) %>%
    mutate(
      Arrest = as.numeric(Arrest),
      Domestic = as.numeric(Domestic)
    ) %>%
    separate(Date, into = c("Date", "Time"), sep = " ", extra = "merge")
}

# Split the data by year and clean each subset
for (year in 2001:2023) {
  yearly_df <- filter(df, lubridate::year(lubridate::mdy_hms(`Date`)) == year)
  
  # Clean the data frame
  cleaned_yearly_df <- clean_data(yearly_df)
  
  # Construct file name for the cleaned data
  cleaned_file_name <- paste0("Cleaned_Crimes_", year, ".csv")
  
  # Write the cleaned data frame to a new CSV file
  write_csv(cleaned_yearly_df, cleaned_file_name)
  
  # Print confirmation message
  print(paste("Written file for year:", year, " - ", cleaned_file_name))
}
