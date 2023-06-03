python run.py -m name=AE_LSVI_visited_cartpole q@q_expl=q_tqrl q@q_eval=q_lcb q_expl.sample_visited_states=True expl_fn=tqrl env=cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=LSVI_cartpole env=cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=AE_LSVI_LCB_cartpole q@q_expl=q_tqrl q@q_eval=q_lcb expl_fn=tqrl env=cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=TQRL_cartpole q@q_expl=q_tqrl q@q_eval=q_eval expl_fn=tqrl env=cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=US_LCB_cartpole q@q_expl=q_us q@q_eval=q_lcb expl_fn=us env=cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=RANDOM_LCB_cartpole q@q_eval=q_lcb q@q_expl=q_random expl_fn=random env=cartpole seed="range(5)" hydra/launcher=joblib
ython run.py -m name=GREEDY_LCB_cartpole q@q_expl=q_eval env=cartpole seed="range(5)" hydra/launcher=joblib
