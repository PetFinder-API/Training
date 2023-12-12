destination_folder="data"

export KAGGLE_CONFIG_DIR=$(pwd)
echo $KAGGLE_CONFIG_DIR
kaggle competitions download -c petfinder-pawpularity-score

# Create the destination folder if it doesn't exist
mkdir -p "$destination_folder"

# Extract the zip file to the destination folder
unzip petfinder-pawpularity-score.zip -d "$destination_folder"

# Print a message indicating that the extraction is complete
echo "Zip file has been extracted to: $destination_folder"

rm petfinder-pawpularity-score.zip