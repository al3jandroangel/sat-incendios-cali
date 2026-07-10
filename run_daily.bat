@echo off
rem Actualizacion del SAT de incendios de Cali (correr 6:00 am y 12:00 m).
rem El MAP_KEY de NASA FIRMS se lee automaticamente de data\firms_key.txt.
rem Programar dos tareas con el Programador de tareas de Windows:
rem   schtasks /create /tn "SAT Incendios Cali 6am"  /tr "D:\Proyectos Personales\SAT Incendios\SAT_Cali\run_daily.bat" /sc daily /st 06:00
rem   schtasks /create /tn "SAT Incendios Cali 12m" /tr "D:\Proyectos Personales\SAT Incendios\SAT_Cali\run_daily.bat" /sc daily /st 12:00
cd /d "D:\Proyectos Personales\SAT Incendios\SAT_Cali\scripts"
python 05_predict_daily.py >> ..\actualizacion.log 2>&1
