filename="enwiki-20100312_0026432145-enwiki-20100312_0005773819-enwiki-20100312_0014504765_target-enwiki-20100312_0005773819_3_4_3_2.p"
orig_test_folder="/home/cc/data/out_salads_70k_Indexed_test2/Test"
geoper_test_folder="/home/cc/data/out_salads_70k_Indexed_GeoPerson_test2/Test"
orig_folder="/home/cc/data/out_salads_70k_Indexed/Test"
geoper_folder="/home/cc/data/out_salads_70k_Indexed_GeoPerson/Test"

rm -r $geoper_test_folder/*
rm -r $orig_test_folder/*
cp $orig_folder/$filename $orig_test_folder
cp $geoper_folder/$filename $geoper_test_folder


