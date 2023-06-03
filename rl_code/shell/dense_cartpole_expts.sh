python run.py -m name=TQRL_LCB_visited_dense_cartpole q@q_expl=q_tqrl q@q_eval=q_lcb q_expl.sample_visited_states=True expl_fn=tqrl env=dense_cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=FOVI_dense_cartpole env=dense_cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=TQRL_LCB_dense_cartpole q@q_expl=q_tqrl q@q_eval=q_lcb expl_fn=tqrl env=dense_cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=US_LCB_dense_cartpole q@q_expl=q_us q@q_eval=q_lcb expl_fn=us env=dense_cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=RANDOM_LCB_dense_cartpole q@q_eval=q_lcb q@q_expl=q_random expl_fn=random env=dense_cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=greedy_dense_cartpole env=dense_cartpole q@q_expl=q_eval seed="range(5)" hydra/launcher=joblib
