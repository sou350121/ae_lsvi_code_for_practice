python run.py -m name=AE_LSVI_visited_uniform_beta_tracking q@q_expl=q_tqrl q@q_eval=q_lcb q_expl.sample_visited_states=True expl_fn=tqrl env=uniform_beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=LSVI_uniform_beta_tracking env=uniform_beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=greedy_uniform_beta_tracking env=uniform_beta_tracking q@q_expl=q_eval seed="range(5)" hydra/launcher=joblib
python run.py -m name=AE_LSVI_uniform_beta_tracking q@q_expl=q_tqrl q@q_eval=q_lcb expl_fn=tqrl env=uniform_beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=US_LCB_uniform_beta_tracking q@q_expl=q_us q@q_eval=q_lcb expl_fn=us env=uniform_beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=RANDOM_LCB_uniform_beta_tracking q@q_eval=q_lcb q@q_expl=q_random expl_fn=random env=uniform_beta_tracking seed="range(5)" hydra/launcher=joblib
