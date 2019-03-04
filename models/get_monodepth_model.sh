#!/bin/bash

model_name=$1
output_location=$2

if [ ! -d $output_location ]; then
	echo "output_location does not exist. Creating it..."
	mkdir -p $output_location
fi

filename=$model_name.zip

url=http://visual.cs.ucl.ac.uk/pubs/monoDepth/models/$filename

output_file=$output_location/$filename

echo "Downloading $model_name"
wget -nc $url -O $output_file
unzip $output_file -d $output_location
rm $output_file