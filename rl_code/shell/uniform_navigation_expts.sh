python run.py -m name=TQRL_LCB_visited_uniform_navigation q@q_expl=q_tqrl q@q_eval=q_lcb q_expl.sample_visited_states=True expl_fn=tqrl env=uniform_navigation seed="range(5)" hydra/launcher=joblib
python run.py -m name=FOVI_uniform_navigation env=uniform_navigation seed="range(5)" hydra/launcher=joblib
python run.py -m name=TQRL_LCB_uniform_navigation q@q_expl=q_tqrl q@q_eval=q_lcb expl_fn=tqrl env=uniform_navigation seed="range(5)" hydra/launcher=joblib
python run.py -m name=TQRL_uniform_navigation q@q_expl=q_tqrl q@q_eval=q_eval expl_fn=tqrl env=uniform_navigation seed="range(5)" hydra/launcher=joblib
python run.py -m name=US_LCB_uniform_navigation q@q_expl=q_us q@q_eval=q_lcb expl_fn=us env=uniform_navigation seed="range(5)" hydra/launcher=joblib
python run.py -m name=RANDOM_LCB_uniform_navigation q@q_eval=q_lcb q@q_expl=q_random expl_fn=random env=uniform_navigation seed="range(5)" hydra/launcher=joblib
python run.py -m name=GREEDY_uniform_navigation q@q_expl=q_eval  env=uniform_navigation seed="range(5)" hydra/launcher=joblib
