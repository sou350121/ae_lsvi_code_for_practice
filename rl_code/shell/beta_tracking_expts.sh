python run.py -m name=LSVI_beta_tracking env=beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=AE_LSVI_beta_tracking q@q_expl=q_tqrl q@q_eval=q_lcb expl_fn=tqrl env=beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=AE_LSVI_visited_beta_tracking q@q_expl=q_tqrl q@q_eval=q_lcb q_expl.sample_visited_states=True expl_fn=tqrl env=beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=US_LCB_beta_tracking q@q_expl=q_us q@q_eval=q_lcb expl_fn=us env=beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=RANDOM_LCB_beta_tracking q@q_eval=q_lcb q@q_expl=q_random expl_fn=random env=beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=greedy_beta_tracking env=beta_tracking q@q_expl=q_eval seed="range(5)" hydra/launcher=joblib
