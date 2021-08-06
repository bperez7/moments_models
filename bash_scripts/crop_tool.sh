#!/bin/bash



while getopts l:t:h:w:f:o:b:e: flag
do
    case "${flag}" in
         f) file=${OPTARG};;
        t) top=${OPTARG};;
        l) left=${OPTARG};;

        h) height=${OPTARG};;
        w) width=${OPTARG};;
       # i) input=${OPTARG};;
        o) output_file=${OPTARG};;
        b) beginning=${OPTARG};;
        e) end=${OPTARG};;


    esac
done
echo "hello";
#echo "height: $height";
#echo "left: $left"
#echo "file $file"
echo "top $top"


ffmpeg -i "$file" -ss "$beginning" -t "$end" -c:v copy -c:a copy trim_ipseek_test.mp4

ffmpeg -i trim_ipseek_test.mp4 -max_muxing_queue_size 1024 -filter:v "crop=${width}:${height}:${left}:${top}" -c:a copy "${output_file}".mp4



#ffmpeg -i videos/IMG_1433.MOV -filter:v "crop=80:60:20:20" -c:a copy out.mp4







#echo "Top: $top";
echo "Left: $left";
echo "Height: $height";

