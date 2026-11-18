import importlib, tabr_all, tabr_noise_experiment as ne
importlib.reload(tabr_all); importlib.reload(ne)
ne.main(["--selftest"])
