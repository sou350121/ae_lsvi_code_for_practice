python run.py -m num_expl_episodes=50 name=AE_LSVI_visited_dense_uniform_cartpole q@q_expl=q_tqrl q@q_eval=q_lcb q_expl.sample_visited_states=True expl_fn=tqrl env=dense_uniform_cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m num_expl_episodes=50 name=greedy_dense_uniform_cartpole env=dense_uniform_cartpole q@q_expl=q_eval seed="range(5)" hydra/launcher=joblib
python run.py -m num_expl_episodes=50 name=LSVI_dense_uniform_cartpole env=dense_uniform_cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m num_expl_episodes=50 name=AE_LSVI_dense_uniform_cartpole q@q_expl=q_tqrl q@q_eval=q_lcb expl_fn=tqrl env=dense_uniform_cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m num_expl_episodes=50 name=US_LCB_dense_uniform_cartpole q@q_expl=q_us q@q_eval=q_lcb expl_fn=us env=dense_uniform_cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m num_expl_episodes=50 name=RANDOM_LCB_dense_uniform_cartpole q@q_eval=q_lcb q@q_expl=q_random expl_fn=random env=dense_uniform_cartpole seed="range(5)" hydra/launcher=joblib
