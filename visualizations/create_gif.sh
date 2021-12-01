# name=$1
# echo ffmpeg -i video/frame%06d.png -r 30 -vf scale=840:525 ${name}.mp4
# echo ffmpeg -i ${name}.mp4 -vf "fps=30,scale=840:525:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" ${name}.gif
# echo xdg-open ${name}.gif
# ffmpeg -i video/frame%06d.png -r 30 -vf scale=840:525 ${name}.mp4
# ffmpeg -i ${name}.mp4 -vf "fps=30,scale=840:525:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" ${name}.gif
# xdg-open ${name}.gif
# cp ${name}.gif ~/disambiguation-colmap/visualizations/

ffmpeg -i video/frame%06d.png -r 30 -vf scale=840:526 $1.mp4
ffmpeg -i $1.mp4 -vf "scale=840:526:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" $1.gif
xdg-open $1.gif
cp $1.gif ~/disambiguation-colmap/visualizations