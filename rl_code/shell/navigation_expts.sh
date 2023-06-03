python run.py -m name=TQRL_LCB_visited_navigation q@q_expl=q_tqrl q@q_eval=q_lcb q_expl.sample_visited_states=True expl_fn=tqrl env=navigation seed="range(5)" hydra/launcher=joblib
python run.py -m name=FOVI_navigation env=navigation seed="range(5)" hydra/launcher=joblib
python run.py -m name=TQRL_LCB_navigation q@q_expl=q_tqrl q@q_eval=q_lcb expl_fn=tqrl env=navigation seed="range(5)" hydra/launcher=joblib
python run.py -m name=TQRL_navigation q@q_expl=q_tqrl q@q_eval=q_eval expl_fn=tqrl env=navigation seed="range(5)" hydra/launcher=joblib
python run.py -m name=US_LCB_navigation q@q_expl=q_us q@q_eval=q_lcb expl_fn=us env=navigation seed="range(5)" hydra/launcher=joblib
python run.py -m name=RANDOM_LCB_navigation q@q_eval=q_lcb q@q_expl=q_random expl_fn=random env=navigation seed="range(5)" hydra/launcher=joblib
python run.py -m name=GREEDY_LCB_navigation q@q_expl=q_eval env=navigation seed="range(5)" hydra/launcher=joblib
