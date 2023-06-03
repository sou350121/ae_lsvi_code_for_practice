python run.py -m name=TQRL_LCB_visited_uniform_beta_rotation q@q_expl=q_tqrl q@q_eval=q_lcb q_expl.sample_visited_states=True expl_fn=tqrl env=uniform_beta_rotation seed="range(5)" hydra/launcher=joblib
python run.py -m name=FOVI_uniform_beta_rotation env=uniform_beta_rotation seed="range(5)" hydra/launcher=joblib
python run.py -m name=greedy_uniform_beta_rotation env=uniform_beta_rotation q@q_expl=q_eval seed="range(5)" hydra/launcher=joblib
python run.py -m name=TQRL_visited_uniform_beta_rotation q@q_expl=q_tqrl q@q_eval=q_eval q_expl.sample_visited_states=True expl_fn=tqrl env=uniform_beta_rotation seed="range(5)" hydra/launcher=joblib
python run.py -m name=TQRL_LCB_uniform_beta_rotation q@q_expl=q_tqrl q@q_eval=q_lcb expl_fn=tqrl env=uniform_beta_rotation seed="range(5)" hydra/launcher=joblib
python run.py -m name=TQRL_uniform_beta_rotation q@q_expl=q_tqrl q@q_eval=q_eval expl_fn=tqrl env=uniform_beta_rotation seed="range(5)" hydra/launcher=joblib
python run.py -m name=US_LCB_uniform_beta_rotation q@q_expl=q_us q@q_eval=q_lcb expl_fn=us env=uniform_beta_rotation seed="range(5)" hydra/launcher=joblib
python run.py -m name=RANDOM_LCB_uniform_beta_rotation q@q_eval=q_lcb q@q_expl=q_random expl_fn=random env=uniform_beta_rotation seed="range(5)" hydra/launcher=joblib
