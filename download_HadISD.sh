#!/bin/bash

# Download all files
printf "Downloading all files...\n"
wget -i HadISD_download_links.txt -P ./data/HadISD/raw/

# Download station metatdata [ID, lat, lon, elev]
wget https://www.metoffice.gov.uk/hadobs/hadisd/v331_202309p/files/hadisd_station_info_v331_202309p.txt -P ./data/HadISD/

# Unzip all folders
printf "Unzipping all folders...\n"
find . -name '*.tar.gz' -execdir tar -xzf '{}' \;

# Unzip individual station files
printf "Unzipping individual station files...\n"
find . -name '*hadisd*.nc.gz' -execdir gunzip '{}' \;