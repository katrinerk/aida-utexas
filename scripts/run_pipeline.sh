wget https://drive.google.com/file/d/1uWbppTp1TE4YvX3euVDpH9PnyLp7_sHG/view?usp=sharing -O /aida-utexas/neural_pipeline/gcn2-cuda_best_5000_1.ckpt
wget https://drive.google.com/file/d/1xQjfWCMrlgiHJvLmEXjUutRFOn_f-Pgu/view?usp=sharing -O /aida-utexas/neural_pipeline/indexers.p
python neural_pipeline/index.py
python neural_pipeline/gen_hypoth.py
