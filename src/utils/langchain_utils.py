def handle_cb(cb):
    cb = cb.__dict__
    del cb['_lock']
    return cb