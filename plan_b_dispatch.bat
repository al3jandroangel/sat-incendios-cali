@echo off
rem Plan B: dispara la actualizacion del SAT en GitHub cada 6 horas.
rem Instalado como tarea programada "SAT Cali Plan B" (ver README).
cd /d "D:\Proyectos Personales\SAT Incendios\SAT_Cali\scripts"
python dispatch_workflow.py >> ..\plan_b.log 2>&1
