python run.py -m num_expl_episodes=34 name=TQRL_LCB_visited_nav_easy q@q_expl=q_tqrl q@q_eval=q_lcb q_expl.sample_visited_states=True expl_fn=tqrl env=nav_easy seed="range(5)" hydra/launcher=joblib
python run.py -m num_expl_episodes=34 name=FOVI_nav_easy env=nav_easy seed="range(5)" hydra/launcher=joblib
python run.py -m num_expl_episodes=34 name=TQRL_LCB_nav_easy q@q_expl=q_tqrl q@q_eval=q_lcb expl_fn=tqrl env=nav_easy seed="range(5)" hydra/launcher=joblib
python run.py -m num_expl_episodes=34 name=US_LCB_nav_easy q@q_expl=q_us q@q_eval=q_lcb expl_fn=us env=nav_easy seed="range(5)" hydra/launcher=joblib
python run.py -m num_expl_episodes=34 name=RANDOM_LCB_nav_easy q@q_eval=q_lcb q@q_expl=q_random expl_fn=random env=nav_easy seed="range(5)" hydra/launcher=joblib
python run.py -m num_expl_episodes=34 name=GREEDY_LCB_nav_easy q@q_expl=q_eval env=nav_easy seed="range(5)" hydra/launcher=joblib
