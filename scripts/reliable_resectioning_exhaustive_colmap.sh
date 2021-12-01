# The path for the colmap compiled with files in reliable_resectioning folder
colmap=
# The path to store the results
results=
# The dataset path
dset=
mkdir ${results}
${colmap} \
  feature_extractor \
  --database_path ${results}/database.db \
  --image_path ${dset}
${colmap} \
  exhaustive_matcher \
  --database_path ${results}/database.db
mkdir ${results}/sparse
${colmap} \
  mapper \
  --database_path ${results}/database.db \
  --image_path ${dset} \
  --output_path ${results}/sparse/
