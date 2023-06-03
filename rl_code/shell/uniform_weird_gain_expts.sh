# python run.py name=AE-LSVI_LCB_visited_uniform_weird_gain q@q_expl=q_tqrl q@q_eval=q_lcb q_expl.sample_visited_states=True expl_fn=tqrl env=uniform_weird_gain eval_fn.num_trials=50
# python run.py  name=LSVI_uniform_weird_gain env=uniform_weird_gain eval_fn.num_trials=50
python run.py name=AE-LSVI_LCB_uniform_weird_gain q@q_expl=q_tqrl q@q_eval=q_lcb expl_fn=tqrl env=uniform_weird_gain
# python run.py -m name=AE-LSVI_LCB_visited_uniform_weird_gain q@q_expl=q_tqrl q@q_eval=q_lcb q_expl.sample_visited_states=True expl_fn=tqrl env=uniform_weird_gain seed="range(5)" hydra/launcher=joblib
# python run.py -m name=LSVI_uniform_weird_gain env=uniform_weird_gain seed="range(5)" hydra/launcher=joblib
# python run.py -m name=greedy_uniform_weird_gain env=uniform_weird_gain q@q_expl=q_eval seed="range(5)" hydra/launcher=joblib
# python run.py -m name=AE-LSVI_visited_uniform_weird_gain q@q_expl=q_tqrl q@q_eval=q_eval q_expl.sample_visited_states=True expl_fn=tqrl env=uniform_weird_gain seed="range(5)" hydra/launcher=joblib
# python run.py -m name=AE-LSVI_LCB_uniform_weird_gain q@q_expl=q_tqrl q@q_eval=q_lcb expl_fn=tqrl env=uniform_weird_gain seed="range(5)" hydra/launcher=joblib
# python run.py -m name=AE-LSVI_uniform_weird_gain q@q_expl=q_tqrl q@q_eval=q_eval expl_fn=tqrl env=uniform_weird_gain seed="range(5)" hydra/launcher=joblib
# python run.py -m name=US_LCB_uniform_weird_gain q@q_expl=q_us q@q_eval=q_lcb expl_fn=us env=uniform_weird_gain seed="range(5)" hydra/launcher=joblib
# python run.py -m name=RANDOM_LCB_uniform_weird_gain q@q_eval=q_lcb q@q_expl=q_random expl_fn=random env=uniform_weird_gain seed="range(5)" hydra/launcher=joblib
