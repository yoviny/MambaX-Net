python -m mambax_net.preprocess.calculate_dataset_fingerprint_segmentation \
 -df ../ProstateCancer-AS/data/marksheet_w_bvalue.csv \
 -o mambax_net/configs/ \
 -i ../ProstateCancer-AS/data/processed/PiccaiData/T2 \
 -wp ../ProstateCancer-AS/data/picai_labels/anatomical_delineations/whole_gland/AI/Guerbet23 \
 -pz ../ProstateCancer-AS/data/picai_labels/anatomical_delineations/zonal_pz_tz/AI/Yuan23 \
 -np 4 -mem 20 -v True --limit 10