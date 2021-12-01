# ffmpeg -framerate $2 -i images/0.jpg -r $2 -vf scale=840:526 $1_dat.mp4
ffmpeg -framerate $2 -pattern_type glob -i "images/*0.jpg" -r $2 -vf scale=840:526 $1_dat.mp4
# ffmpeg -framerate $2 -pattern_type glob -i "images/*.jpg" -r $2 -vf scale=840:526 $1_dat.mp4
ffmpeg -i $1_dat.mp4 $1_dat.gif
xdg-open $1_dat.gif
cp $1_dat.gif ~/disambiguation-colmap/visualizations/