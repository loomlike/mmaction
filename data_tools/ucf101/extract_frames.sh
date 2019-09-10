#! /usr/bin/bash env

cd ../
python build_rawframes.py ../data/ucf101/videos/ ../data/ucf101/rawframes/ --level 2 --flow_type tvl1 --df_path ../third_party/dense_flow --num_gpu 4
echo "Raw frames (RGB and tv-l1) Generated"
cd ucf101/
