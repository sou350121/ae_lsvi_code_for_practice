python run.py -m num_expl_episodes=34 name=TQRL_LCB_visited_uniform_nav_easy q@q_expl=q_tqrl q@q_eval=q_lcb q_expl.sample_visited_states=True expl_fn=tqrl env=uniform_nav_easy seed="range(5)" hydra/launcher=joblib
python run.py -m num_expl_episodes=34 name=FOVI_uniform_nav_easy env=uniform_nav_easy seed="range(5)" hydra/launcher=joblib
python run.py -m num_expl_episodes=34 name=TQRL_LCB_uniform_nav_easy q@q_expl=q_tqrl q@q_eval=q_lcb expl_fn=tqrl env=uniform_nav_easy seed="range(5)" hydra/launcher=joblib
python run.py -m num_expl_episodes=34 name=US_LCB_uniform_nav_easy q@q_expl=q_us q@q_eval=q_lcb expl_fn=us env=uniform_nav_easy seed="range(5)" hydra/launcher=joblib &
python run.py -m num_expl_episodes=34 name=RANDOM_LCB_uniform_nav_easy q@q_eval=q_lcb q@q_expl=q_random expl_fn=random env=uniform_nav_easy seed="range(5)" hydra/launcher=joblib &
 python run.py -m num_expl_episodes=34 name=GREEDY_LCB_uniform_nav_easy q@q_expl=q_eval env=uniform_nav_easy seed="range(5)" hydra/launcher=joblib
