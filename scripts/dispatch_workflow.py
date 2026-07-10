# -*- coding: utf-8 -*-
"""Plan B de actualización: dispara el workflow de GitHub por API.

Los crons de GitHub Actions pueden retrasarse u omitirse; este script (que la
tarea programada de Windows ejecuta cada 6 horas) garantiza el disparo. Usa el
token de GitHub guardado en el Administrador de credenciales de Windows.
"""
import datetime as dt
import json
import subprocess
import urllib.request

REPO = "al3jandroangel/sat-incendios-cali"


def token():
    out = subprocess.run(["git", "credential", "fill"],
                         input="protocol=https\nhost=github.com\n\n",
                         capture_output=True, text=True).stdout
    return dict(l.split("=", 1) for l in out.strip().splitlines())["password"]


def main():
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/actions/workflows/"
        "actualizar-alertas.yml/dispatches",
        data=json.dumps({"ref": "main"}).encode(),
        headers={"Authorization": f"Bearer {token()}",
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "sat-cali-planb"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        print(f"{dt.datetime.now():%Y-%m-%d %H:%M} disparo Plan B -> {r.status}")


if __name__ == "__main__":
    main()
