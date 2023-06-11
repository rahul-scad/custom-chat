

import logging
from modules import shared


host = shared.args.listen_host if shared.args.listen_host and shared.args.listen else '127.0.0.1'
port = shared.args.listen_port if shared.args.listen_port else '7860'

options = {
    'addr': f"{host}:{port}",
    'authtoken_from_env': True,
    'session_metadata': 'text-generation-webui',
}

def ui():
    settings = shared.settings.get("ngrok")
    if settings:
        options.update(settings)

    try:
        import ngrok
        tunnel = ngrok.connect(**options)
        logging.info(f"Ingress established at: {tunnel.url()}")
    except ModuleNotFoundError:
        logging.error("===> ngrok library not found, please run `pip install -r extensions/ngrok/requirements.txt`")

