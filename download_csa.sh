# wget --cut-dirs=7 -P ./ -nH -np -m ftp://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data
wget --cut-dirs=8 -P ./ -nH -np -m https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/
rm csa_data/raw_data *index*
rm csa_data/raw_data/ *index*
rm csa_data/raw_data/*index*
rm csa_data/raw_data/splits/*index*
rm csa_data/raw_data/x_data/*index*
rm csa_data/raw_data/y_data/*index*
rm csa_data/*index*