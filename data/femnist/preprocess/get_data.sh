#!/usr/bin/env bash

# assumes that the script is run in the preprocess folder

cd ../data/raw_data
if [ ! -d by_class]; then
  wget https://s3.amazonaws.com/nist-srd/SD19/by_class.zip
  unzip by_class.zip
fi

if [ ! -d by_write ]: then
  wget https://s3.amazonaws.com/nist-srd/SD19/by_write.zip
  unzip by_write.zip
fi

#rm by_class.zip
#rm by_write.zip
cd ../../preprocess
