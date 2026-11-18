s = open('tabr_all.py').read()
if 'return_model' not in s:
    a = 'def run_one(data, seed, args, device, name=""):'
    b = '    res["time_s"] = time.time() - t0\n    return res'
    assert a in s and b in s, "pattern not found - patch run_one by hand"
    s = s.replace(a, 'def run_one(data, seed, args, device, name="", return_model=False):')
    s = s.replace(b, '    res["time_s"] = time.time() - t0\n    if return_model:\n        res.update(_model=model, _tr=tr, _eval_bs=eval_bs)\n    return res')
    open('tabr_all.py', 'w').write(s)
print("patched:", 'return_model' in open('tabr_all.py').read())
