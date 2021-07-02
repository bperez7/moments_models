#!/bin/bash



while getopts l:t:h:w:f:o: flag
do
    case "${flag}" in
         f) file=${OPTARG};;
        t) top=${OPTARG};;
        l) left=${OPTARG};;

        h) height=${OPTARG};;
        w) width=${OPTARG};;
        o) output_file=${OPTARG};;


    esac
done
echo "hello";
#echo "height: $height";
#echo "left: $left"
#echo "file $file"
echo "top $top"


ffmpeg -i "$file" -filter:v "crop=${width}:${height}:${left}:${top}" -c:a copy "${output_file}".mp4
#ffmpeg -i videos/IMG_1433.MOV -filter:v "crop=80:60:20:20" -c:a copy out.mp4







#echo "Top: $top";
echo "Left: $left";
echo "Height: $height";

