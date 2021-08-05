#!/bin/bash
input="Output.txt"
while read line;
do
  bash crop_tool.sh line
  #bash crop_tool.sh
done < "$input"

